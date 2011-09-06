from twisted.internet.defer import inlineCallbacks
from twisted.internet.task import LoopingCall

from bravo.blocks import blocks, items, furnace_fuel, unstackable
from bravo.inventory import Slot
from bravo.inventory.windows import FurnaceWindow

# TODO: move this out of the module into plug-in
furnace_recipes = {
    blocks["gold-ore"].slot : Slot(items["gold-ingot"].slot, 0, 1),
    blocks["iron-ore"].slot : Slot(items["iron-ingot"].slot, 0, 1),
    blocks["diamond-ore"].slot : Slot(items["diamond"].slot, 0, 1),
    blocks["log"].slot : Slot(items["coal"].slot, 1, 1), # charcoal
    blocks["cactus"].slot : Slot(items["dye"].slot, 2, 1), # green dye
    blocks["sand"].slot : Slot(blocks["glass"].slot, 0, 1),
    blocks["cobblestone"].slot : Slot(blocks["stone"].slot, 0, 1),
    items["clay-balls"].slot : Slot(items["clay-brick"].slot, 0, 1),
    items["raw-porkchop"].slot : Slot(items["cooked-porkchop"].slot, 0, 1),
    items["raw-fish"].slot : Slot(items["cooked-fish"].slot, 0, 1)
}

class FurnaceManager(object):

    def __init__(self, factory):
        self.factory = factory
        self.furnaces = {}
        self.cleanup_timer = LoopingCall(self.cleanup)

    def start(self):
        """
        Enable this manager.

        While this manager is running, furnaces will be reaped every 5
        minutes.
        """

        self.cleanup_timer.start(300)

    def stop(self):
        self.cleanup_timer.stop()

    @inlineCallbacks
    def update(self, coords):
        # We've got informed that furnace content is changed
        if coords not in self.furnaces:
            bigx, smallx, bigz, smallz, y = coords
            chunk = yield self.factory.world.request_chunk(bigx, bigz)
            tile = chunk.tiles[(smallx, y, smallz)]
            self.furnaces[coords] = FurnaceProcess(tile, coords)
            self.furnaces[coords].factory = self.factory
        self.furnaces[coords].update()

    def remove(self, coords):
        if coords in self.furnaces:
            del(self.furnaces[coords])

    def cleanup(self):
        # remove processes that do not run
        for c in self.furnaces.keys():
            if not self.furnaces[c].running:
                self.remove(c)

class FurnaceProcess(object):
    '''
    NOTE: Our furnace process doesn't operate with world ticks.
          We do updates twice per second. It's our UI update rate.
    '''
    def __init__(self, tile, coords):
        self.tile = tile
        self.coords = coords
        self.running = False
        self.burning = LoopingCall(self.burn)

    def update(self):
        if not self.running:
            if self.hasFuel and self.canCraft:
                self.tile.burntime = 0
                self.tile.cooktime = 0
                self.burning.start(0.5) # start burning loop

    def burn(self):
        # -----------------------------
        # ---     item crafting     ---
        # -----------------------------
        if self.canCraft:
            self.tile.cooktime += 1
            # Notchian time is ~9.25-9.50 sec.
            if self.tile.cooktime == 20: # cooked!
                source = self.tile.inventory.crafting[0]
                product = furnace_recipes[source.primary]
                self.tile.inventory.crafting[0] = source.decrement()
                if self.tile.inventory.crafted[0] is None:
                    self.tile.inventory.crafted[0] = product
                else:
                    item = self.tile.inventory.crafted[0]
                    self.tile.inventory.crafted[0] = item.increment(product.quantity)
                self.update_all_windows_slot(0, self.tile.inventory.crafting[0])
                self.update_all_windows_slot(2, self.tile.inventory.crafted[0])
                self.tile.cooktime = 0
        else:
            self.tile.cooktime = 0

        # ----------------------------
        # ---     fuel consume     ---
        # ----------------------------
        if self.tile.burntime == 0:
            if self.hasFuel and self.canCraft: # burn next portion of the fuel
                fuel = self.tile.inventory.fuel[0]
                self.tile.burntime = self.burn_max = furnace_fuel[fuel.primary]
                self.tile.inventory.fuel[0] = fuel.decrement()
                if not self.running:
                    self.on_off(True)
                self.update_all_windows_slot(1, self.tile.inventory.fuel[0])
            else: # out of fuel or no need to burn more
                self.burning.stop()
                self.on_off(False)
                # reset cook time
                self.tile.cooktime = 0
                self.update_all_windows_progress(0, 0)
                return
        self.tile.burntime -= 1

        # ----------------------------
        # --- update progress bars ---
        # ----------------------------
        cook_progress = 185 * self.tile.cooktime / 19
        burn_progress = 250 * self.tile.burntime / self.burn_max
        self.update_all_windows_progress(0, cook_progress)
        self.update_all_windows_progress(1, burn_progress)

    def on_off(self, state):
        self.running = state
        bigx, smallx, bigz, smallz, y = self.coords
        block = state and blocks["burning-furnace"] or blocks["furnace"]
        d = self.factory.world.request_chunk(bigx, bigz)
        @d.addCallback
        def replace_furnace_block(chunk):
            chunk.set_block((smallx, y, smallz), block.slot)
            self.factory.flush_chunk(chunk)

    def update_all_windows_slot(self, slot, item):
        # update all opened windows
        for p in self.factory.protocols.itervalues():
            if p.windows and type(p.windows[-1]) == FurnaceWindow:
                window = p.windows[-1]
                if window.coords == self.coords:
                    if item is None:
                        p.write_packet("window-slot",
                            wid=window.wid, slot=slot, primary=-1)
                    else:
                        p.write_packet("window-slot",
                            wid=window.wid, slot=slot, primary=item.primary,
                            secondary=item.secondary, count=item.quantity)

    def update_all_windows_progress(self, bar, value):
        # update all opened windows
        for p in self.factory.protocols.itervalues():
            if p.windows and type(p.windows[-1]) == FurnaceWindow:
                window = p.windows[-1]
                if window.coords == self.coords:
                    p.write_packet("window-progress", wid=window.wid,
                        bar=bar, progress=value)

    @property
    def hasFuel(self):
        # if the furnace hase something to burn
        if self.tile.inventory.fuel[0] is None:
            return False
        else:
            return self.tile.inventory.fuel[0].primary in furnace_fuel

    @property
    def canCraft(self):
        # if have somethig to craft from...
        if self.tile.inventory.crafting[0] is None:
            return False
        if self.tile.inventory.crafting[0].primary in furnace_recipes:
            #...and has space for it
            if self.tile.inventory.crafted[0] is None:
                return True
            else:
                crafting = self.tile.inventory.crafting[0]
                crafted = self.tile.inventory.crafted[0]
                if furnace_recipes[crafting.primary][0] != crafted.primary:
                    return False
                elif crafted.primary in unstackable:
                    return False
                elif crafted.quantity + furnace_recipes[crafting.primary].quantity > 64:
                    return False
                else:
                    return True
        else:
            return False
