from array import array
from itertools import product
from warnings import warn

from twisted.internet.defer import maybeDeferred

from bravo.blocks import blocks, glowing_blocks
from bravo.entity import Mob
from bravo.packets.beta import make_packet
from bravo.utilities.bits import pack_nibbles

class ChunkWarning(Warning):
    """
    Somebody did something inappropriate to this chunk, but it probably isn't
    lethal, so the chunk is issuing a warning instead of an exception.
    """

def clamp(x, low, high):
    return min(max(x, low), high)

# Set up glow tables.
# These tables provide glow maps for illuminated points.
glow = [None] * 16
for i in range(16):
    dim = 2 * i + 1
    glow[i] = array("b", [0] * (dim**3))
    for x, y, z in product(xrange(dim), repeat=3):
        distance = abs(x - i) + abs(y - i) + abs(z - i)
        glow[i][(x * dim + y) * dim + z] = i + 1 - distance
    glow[i] = array("B", [clamp(x, 0, 15) for x in glow[i]])

def composite_glow(target, strength, x, y, z):
    """
    Composite a light source onto a lightmap.

    The exact operation is not quite unlike an add.
    """

    ambient = glow[strength]

    xbound, zbound, ybound = target.shape

    sx = x - strength
    sy = y - strength
    sz = z - strength

    ex = x + strength
    ey = y + strength
    ez = z + strength

    si, sj, sk = 0, 0, 0
    ei, ej, ek = strength * 2, strength * 2, strength * 2

    if sx < 0:
        sx, si = 0, -sx

    if sy < 0:
        sy, sj = 0, -sy

    if sz < 0:
        sz, sk = 0, -sz

    if ex > xbound:
        ex, ei = xbound, ei - ex + xbound

    if ey > ybound:
        ey, ej = ybound, ej - ey + ybound

    if ez > zbound:
        ez, ek = zbound, ek - ez + zbound

    # Composite!
    target[sx:ex, sz:ez, sy:ey] += ambient[si:ei, sk:ek, sj:ej]

class Chunk(object):
    """
    A chunk of blocks.

    Chunks are large pieces of world geometry (block data). The blocks, light
    maps, and associated metadata are stored in chunks. Chunks are
    always measured 16x128x16 and are aligned on 16x16 boundaries in
    the xz-plane.

    :cvar bool dirty: Whether this chunk needs to be flushed to disk.
    :cvar bool populated: Whether this chunk has had its initial block data
        filled out.
    """

    dirty = True
    populated = False

    def __init__(self, x, z):
        """
        :param int x: X coordinate in chunk coords
        :param int z: Z coordinate in chunk coords

        :ivar numpy.ndarray heightmap: Tracks the tallest block in each xz-column.
        :ivar numpy.ndarray skylight: Ambient light map.
        :ivar numpy.ndarray damaged: Array for tracking damaged coordinates.
        :ivar bool all_damaged: Flag for forcing the entire chunk to be
            damaged. This is for efficiency; past a certain point, it is not
            efficient to batch block updates or track damage. Heavily damaged
            chunks have their damage represented as a complete resend of the
            entire chunk.
        """

        self.x = int(x)
        self.z = int(z)

        self.blocks = array("B", [0] * (16 * 16 * 128))
        self.heightmap = array("B", [0] * (16 * 16))
        self.blocklight = array("B", [0] * (16 * 16 * 128))
        self.metadata = array("B", [0] * (16 * 16 * 128))
        self.skylight = array("B", [0] * (16 * 16 * 128))

        self.entities = set()
        self.tiles = {}

        self.damaged = array("B", [0] * (16 * 16 * 128))

        self.all_damaged = False

    def __repr__(self):
        return "Chunk(%d, %d)" % (self.x, self.z)

    __str__ = __repr__

    def regenerate_heightmap(self):
        """
        Regenerate the height map array.

        The height map is merely the position of the tallest block in any
        xz-column.
        """

        for x, z in product(xrange(16), repeat=2):
            for y in range(127, -1, -1):
                if self.blocks[(x * 16 + z) * 16 + y]:
                    break

            self.heightmap[x * 16 + z] = y

    def regenerate_blocklight(self):
        lightmap = array("L", [0] * (16 * 16 * 128))

        for x, y, z in product(xrange(16), xrange(128), xrange(16)):
            block = self.blocks[x, z, y]
            if block in glowing_blocks:
                composite_glow(lightmap, glowing_blocks[block], x, y, z)

        self.blocklight = cast["uint8"](lightmap.clip(0, 15))

    def regenerate_metadata(self):
        pass

    def regenerate_skylight(self):
        """
        Regenerate the ambient light map.

        Each block's individual light comes from two sources. The ambient
        light comes from the sky.

        The height map must be valid for this method to produce valid results.
        """

        lightmap = zeros((16, 16, 128), dtype="uint8")

        for x, z in product(xrange(16), repeat=2):
            # The maximum lighting value, unsurprisingly, is 0xf, which is the
            # biggest possible value for a nibble.
            light = 0xf

            # Apparently, skylights start at the block *above* the block on
            # which the light is incident?
            height = self.heightmap[x, z] + 1

            # The topmost block, regardless of type, is set to maximum
            # lighting, as are all the blocks above it.
            lightmap[x, z, height:] = light

            # Dim the light going throught the remaining blocks, until there
            # is no more light left.
            for y in range(height, -1, -1):
                dim = blocks[self.blocks[x, z, y]].dim
                light -= dim
                if light <= 0:
                    break

                lightmap[x, z, y] = light

        # Now it's time to spread the light around. This flavor uses extra
        # memory to speed things up; the basic idea is to spread *all* light,
        # one glow level at a time, rather than spread each block
        # individually.
        max_height = amax(self.heightmap)
        lightable = vectorize(lambda block: blocks[block].dim < 15)(self.blocks)
        # Protip: This is a bitwise AND because logical ANDs on arrays can't
        # happen in Numpy.
        unlighted = logical_not(lightmap) & lightable

        # Create a mask to find all blocks that have an unlighted block
        # as a neighbour in the xz-plane.
        mask = zeros((16, 16, max_height), dtype="bool")
        mask[:-1,:,:max_height] |= unlighted[1:, :, :max_height]
        mask[:,:-1,:max_height] |= unlighted[:, 1:, :max_height]
        mask[1:,:,:max_height] |= unlighted[:-1, :, :max_height]
        mask[:,1:,:max_height] |= unlighted[:, :-1, :max_height]

        # Apply the mask to the lightmap to find all lighted blocks with one
        # or more unlighted blocks as neighbours.
        edges = logical_and(mask, lightmap[:, :, :max_height]).nonzero()

        spread = [tuple(coords) for coords in transpose(edges)]
        visited = set()

        # Run the actual glow loop. For each glow level, go over unvisited air
        # blocks and illuminate them.
        for glow in range(14, 0, -1):
            for coords in spread:
                if lightmap[coords] <= glow:
                    visited.add(coords)
                    continue

                for dx, dz, dy in (
                    (1, 0, 0),
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, -1, 0),
                    (0, 0, 1),
                    (0, 0, -1)):
                    x, z, y = coords
                    x += dx
                    z += dz
                    y += dy

                    if not (0 <= x < 16 and
                        0 <= z < 16 and
                        0 <= y < 128):
                        continue

                    if (x, z, y) in visited:
                        continue

                    if lightable[x, z, y] and lightmap[x, z, y] < glow:
                        lightmap[x, z, y] = glow - blocks[self.blocks[x, z, y]].dim
                        visited.add((x, z, y))
            spread = visited
            visited = set()

        self.skylight = lightmap.clip(0, 15)

    def regenerate(self):
        """
        Regenerate all extraneous tables.
        """

        self.regenerate_heightmap()
        self.regenerate_blocklight()
        self.regenerate_metadata()
        self.regenerate_skylight()

        self.dirty = True

    def damage(self, coords):
        """
        Record damage on this chunk.
        """

        if self.all_damaged:
            return

        x, y, z = coords

        self.damaged[(x * 16 + z) * 16 + y] = 1

        if sum(self.damaged) > 176:
            self.all_damaged = True

    def is_damaged(self):
        """
        Determine whether any damage is pending on this chunk.

        :rtype: bool
        :returns: True if any damage is pending on this chunk, False if not.
        """

        return self.all_damaged or any(self.damaged)

    def get_damage_packet(self):
        """
        Make a packet representing the current damage on this chunk.

        This method is not private, but some care should be taken with it,
        since it wraps some fairly cryptic internal data structures.

        If this chunk is currently undamaged, this method will return an empty
        string, which should be safe to treat as a packet. Please check with
        `is_damaged()` before doing this if you need to optimize this case.

        To avoid extra overhead, this method should really be used in
        conjunction with `Factory.broadcast_for_chunk()`.

        Do not forget to clear this chunk's damage! Callers are responsible
        for doing this.

        >>> packet = chunk.get_damage_packet()
        >>> factory.broadcast_for_chunk(packet, chunk.x, chunk.z)
        >>> chunk.clear_damage()

        :rtype: str
        :returns: String representation of the packet.
        """

        if self.all_damaged:
            # Resend the entire chunk!
            return self.save_to_packet()
        elif not any(self.damaged):
            # Send nothing at all; we don't even have a scratch on us.
            return ""
        elif sum(self.damaged) == 1:
            # Use a single block update packet. Find the first (only) set bit
            # in the damaged array, and use it as an index.
            index = next(i for i, value in enumerate(self.damaged) if value)
            # divmod() trick for coords.
            index, y = divmod(index, 128)
            x, z = divmod(index, 16)

            return make_packet("block",
                    x=x + self.x * 16,
                    y=y,
                    z=z + self.z * 16,
                    type=self.blocks[index],
                    meta=self.metadata[index])
        else:
            # Use a batch update.
            # Coordinates are not quite packed in the same system as the
            # indices for chunk data structures.
            # Chunk data structures are ((x * 16) + z) * 128) + y, or in
            # bit-twiddler's parlance, x << 11 | z << 7 | y. However, for
            # this, we need x << 12 | z << 8 | y, so repack accordingly.
            coords = []
            types = []
            metadata = []
            for index, value in self.damaged:
                if value:
                    # This line deserves an explanation. The top of index is
                    # correct, but needs to be repacked. x and z are 4 bits
                    # wide, and need to be one bit higher, so we mask them
                    # together and shift them both up, while preserving the y.
                    repacked = ((index & 0x7f80) << 1) | (index & 0x7f)
                    coords.append(repacked)
                    types.append(self.blocks[index])
                    metadata.append(self.metadata[index])

            return make_packet("batch", x=self.x, z=self.z,
                length=len(coords), coords=coords, types=types,
                metadata=metadata)

    def clear_damage(self):
        """
        Clear this chunk's damage.
        """

        self.damaged = array("B", [0] * (16 * 16 * 128))
        self.all_damaged = False

    def save_to_packet(self):
        """
        Generate a chunk packet.
        """

        array = self.blocks.tostring()
        array += pack_nibbles(self.metadata)
        array += pack_nibbles(self.blocklight)
        array += pack_nibbles(self.skylight)
        packet = make_packet("chunk", x=self.x * 16, y=0, z=self.z * 16,
            x_size=15, y_size=127, z_size=15, data=array)
        return packet

    def get_block(self, coords):
        """
        Look up a block value.

        :param tuple coords: coordinate triplet
        :rtype: int
        :returns: int representing block type
        """

        try:
            x, y, z = coords
            return self.blocks[(x * 16 + z) * 16 + y]
        except IndexError:
            # Coordinates were out-of-bounds; warn and pretend it's air.
            warn("Coordinates %s are out-of-bounds in %s" % (coords, self),
                 ChunkWarning)
            return 0

    def set_block(self, coords, block):
        """
        Update a block value.

        :param tuple coords: coordinate triplet
        :param int block: block type
        """

        x, y, z = coords

        try:
            if self.blocks[x, z, y] != block:
                self.blocks[x, z, y] = block

                if not self.populated:
                    return

                # Regenerate heightmap at this coordinate.
                if not block:
                    # If we replace the highest block with air, we need to go
                    # through all blocks below it to find the new top block.
                    height = self.heightmap[x, z]
                    if y == height:
                        for y in range(height, -1, -1):
                            if self.blocks[x, z, y]:
                                break
                        self.heightmap[x, z] = y
                else:
                    self.heightmap[x, z] = max(self.heightmap[x, z], y)

                # Add to lightmap at this coordinate.
                if block in glowing_blocks:
                    composite_glow(self.blocklight, glowing_blocks[block],
                        x, y, z)

                    self.blocklight = cast["uint8"](self.blocklight.clip(0, 15))

                self.dirty = True
                self.damage(coords)
        except IndexError:
            # Coordinates were out-of-bounds; warn and run away.
            warn("Coordinates %s are out-of-bounds in %s" % (coords, self),
                 ChunkWarning)

    def get_metadata(self, coords):
        """
        Look up metadata.

        :param tuple coords: coordinate triplet
        :rtype: int
        """

        x, y, z = coords

        try:
            return self.metadata[x, z, y]
        except IndexError:
            # Coordinates were out-of-bounds; warn.
            warn("Coordinates %s are out-of-bounds in %s" % (coords, self),
                 ChunkWarning)
            return 0

    def set_metadata(self, coords, metadata):
        """
        Update metadata.

        :param tuple coords: coordinate triplet
        :param int metadata:
        """

        x, y, z = coords

        try:
            if self.metadata[x, z, y] != metadata:
                self.metadata[x, z, y] = metadata

                self.dirty = True
                self.damage(coords)
        except IndexError:
            # Coordinates were out-of-bounds; warn.
            warn("Coordinates %s are out-of-bounds in %s" % (coords, self),
                 ChunkWarning)

    def destroy(self, coords):
        """
        Destroy the block at the given coordinates.

        This may or may not set the block to be full of air; it uses the
        block's preferred replacement. For example, ice generally turns to
        water when destroyed.

        This is safe as a no-op; for example, destroying a block of air with
        no metadata is not going to cause state changes.

        :param tuple coords: coordinate triplet
        """

        x, y, z = coords

        block = blocks[self.blocks[x, z, y]]
        self.set_block((x, y, z), block.replace)
        self.set_metadata((x, y, z), 0)

    def height_at(self, x, z):
        """
        Get the height of an xz-column of blocks.

        :param int x: X coordinate
        :param int z: Z coordinate
        :rtype: int
        :returns: The height of the given column of blocks.
        """

        return self.heightmap[x, z]

    def sed(self, search, replace):
        """
        Execute a search and replace on all blocks in this chunk.

        Named after the ubiquitous Unix tool. Does a semantic
        s/search/replace/g on this chunk's blocks.

        :param int search: block to find
        :param int replace: block to use as a replacement
        """

        results = self.blocks == search

        if results.any():
            self.all_damaged = True
            self.dirty = True

            self.blocks = where(results, replace, self.blocks)

    def get_column(self, x, z):
        """
        Return a slice of the block data at the given xz-column.

        The slice is a numpy array, so you do not have to set it again if you
        are modifying it in-place.

        :rtype: :py:class:`numpy.ndarray`
        """
        return self.blocks[x, z]

    def set_column(self, x, z, column):
        """
        Atomically set an entire xz-column's block data.

        :param int x: X coordinate
        :param int z: Z coordinate
        :type column: :py:class:`numpy.ndarray`
        :param column: Column data, in the form of a NumPy array.
        """
        self.blocks[x, z] = column

        self.dirty = True
        for y in range(128):
            self.damage((x, y, z))

    def update_entities(self, factory):
        """
        Request that the provided factory update this chunk's entities.
        """

        # XXX this method is really bad inversion of control

        for entity in self.entities:
            # XXX bad polymorphism
            if isinstance(entity, Mob):
                # XXX um, WTF. Why is this here?
                maybeDeferred(entity.update_location, factory)
