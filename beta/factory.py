from twisted.internet.protocol import Factory
from twisted.internet.task import LoopingCall
from twisted.plugin import getPlugins

from beta.ibeta import IAuthenticator
import beta.plugins
from beta.protocol import AlphaProtocol
from beta.world import World

(STATE_UNAUTHENTICATED, STATE_CHALLENGED, STATE_AUTHENTICATED,
    STATE_LOCATED) = range(4)

authenticator = "offline"

class AlphaFactory(Factory):

    protocol = AlphaProtocol

    def __init__(self):
        self.world = World("world")
        self.players = set()

        self.entityid = 1
        self.entities = set()

        self.time = 0
        self.time_loop = LoopingCall(self.update_time)
        self.time_loop.start(10)

        self.hooks = {}

        print "Discovering authenticators..."
        authenticators = [i for i in getPlugins(IAuthenticator, beta.plugins)]
        for plugin in authenticators:
            print " ~ Plugin: %s" % plugin.name

        if len(authenticators) == 1:
            selected = authenticators[0]
        else:
            selected = next(i for i in authenticators
                if i.name == authenticator)
            if not selected:
                selected = authenticators[0]

        print "Using authenticator %s" % selected.name
        self.hooks[2] = selected.handshake
        self.hooks[1] = selected.login

        print "Factory init'd"

    def create_entity(self, x = 0, y = 0, z = 0, entity_type = None):
        self.entityid += 1
        entity = Entity(self.entityid, x, y, z, entity_type)
        self.entities.add(entity)
        return entity

    def destroy_entity(self, entity):
        self.entities.discard(entity)

    def update_time(self):
        self.time += 200
        while self.time > 24000:
            self.time -= 24000

    def broadcast(self, packet):
        for player in self.players:
            player.transport.write(packet)

    def broadcast_for_chunk(self, packet, x, z):
        """
        Broadcast a packet to all players that have a certain chunk loaded.

        `x` and `z` are chunk coordinates, not block coordinates.
        """

        for player in self.players:
            if (x, z) in player.chunks:
                player.transport.write(packet)

    def entities_near(self, x, y, z, radius):
        """
        Given a coordinate and a radius, return all entities within that
        radius of those coordinates.

        All arguments should be in pixels, not blocks.

        This method is lazy.
        """

        for entity in self.entities:
            if ((entity.x - x)**2 + (entity.y - y)**2 + (entity.z - z)**2 <
                radius):
                yield entity

class Entity(object):
    def __init__(self, id, x = 0, y = 0, z = 0, entity_type = None):
        self.id, self.x, self.y, self.z, self.entity_type = id,x,y,z,entity_type
