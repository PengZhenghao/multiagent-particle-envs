import numpy as np
from skimage.measure import label

ENCOUNTER_DIFFERENT_CLASS_AGENTS = 100


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        self.track = []

        # This three list should have equal length.
        self.position = []
        self.action = []
        self.communication = []
        # self.current_location_index = 1
        self.class_index = None  # Start from 1


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # state
        self.state = EntityState()


# properties of agent entities
class Agent(Entity):
    def __init__(self, ind=None, class_ind=None):
        assert ind is not None
        # assert class_ind is not None
        self.class_index = class_ind
        self.index = ind
        self.name = 'agent %d' % ind
        super(Agent, self).__init__()
        # cannot send communication signals
        self.silent = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # state
        self.state = AgentState()
        # action
        self.action = Action()


# multi-agent world
class World(object):
    def __init__(self, m, class_list, list_of_agents_loc, shape, silent):
        # list of agents and entities (can change at execution-time!)
        self.class_list = class_list
        self.agents = []
        self.shape = np.array(shape)  # [x, y, nlayers]
        self.map = m
        self.silent = silent
        self.encounter_flags = None
        # self.drawer =

        tmp_index_of_agent = 0
        for class_ind, n_agents in enumerate(class_list):
            # assert n_agents==len(list_of_agents_loc[class_ind])
            for i in range(n_agents):
                x, y = list_of_agents_loc[class_ind][i]
                tmp_agent = Agent(ind=tmp_index_of_agent, class_ind=class_ind + 1)
                tmp_agent.state.p_pos = np.array([x, y, 0])
                tmp_agent.state.track.append(np.array([x, y, 0]))
                tmp_agent.state.position.append(np.array([x, y, 0]))
                tmp_agent.silent = self.silent
                self.agents.append(tmp_agent)
                tmp_index_of_agent += 1

        # communication channel dimensionality
        self.dim_c = 1
        # position dimensionality
        self.dim_p = 3
        self.n_iterations = 0

    # update state of the world, return nothing
    def step(self):
        self.encounter_flags = [{'same': False, 'diff': False, 'leave_same': False, 'leave_diff': False} for _ in
                                range(len(self.agents))]  # encounter_same_class(connect), encounter_diff_class
        for agent_ind, agent in enumerate(self.agents):
            self.step_each(agent, agent_ind)
        self.n_iterations += 1
        return self.encounter_flags

    def step_each(self, agent, agent_ind):

        # Undo
        if agent.action.u == 0:  # 动作为0，原地不动
            return
        if agent.action.u == 7 is None:  # Undo Operation
            self.step_each_undo(agent, agent_ind)
        # Other
        else:
            self.step_each_else(agent, agent_ind)
            # return encounter_flags

    def step_each_undo(self, agent, agent_ind):
        pos = agent.state.p_pos
        agents = self.agents
        if len(agent.state.track) == 1:  # When agent is in the initial position.
            return
        del agent.state.track[-1]
        # agent.state.current_location_index -= 1
        agent.state.p_pos = agent.state.track[-1]

        # To deal with the situation that an agent has walked by a position multiple times.
        # For convenience,
        flag = True
        for ind, any_pos in zip(list(range(len(agent.state.track), 0, -1)), agent.state.track[::-1]):
            # assert np.array_equal(agent.state.track[ind], any_pos)
            if np.array_equal(any_pos, pos):  # Retrieve previous step's index if walked this place before.
                self.map[agent.index, pos[0], pos[1], pos[2]] = ind  # since self.agent.current_loc_ind begin with 1
                flag = False
                break
        if flag:
            self.map[agent.index, pos[0], pos[1], pos[2]] = 0

        # To deal with the situation that multiple agents have walked by the same position.
        overlap = False
        tmp_class_of_agent = None
        self.map[-1, pos[0], pos[1], pos[2]] = 0
        for ind, m in enumerate(self.map[:-1]):
            if m[pos[0], pos[1], pos[2]] != 0:
                # Somebody stay here, you leave them, mean either you leave your pal or your enemy.
                if agents[ind].class_index != agent.class_index:
                    self.encounter_flags[agent_ind]['leave_diff'] = True
                else:
                    self.encounter_flags[agent_ind]['leave_same'] = True
                if overlap:
                    if tmp_class_of_agent == agents[ind].class_index:  # Encounter agent in the same class
                        pass
                        # self.map[-1, pos[0], pos[1], pos[2]] = m[pos[0], pos[1], pos[2]]
                        # encounter_same_class = True
                    else:
                        self.map[-1, pos[0], pos[1], pos[2]] = ENCOUNTER_DIFFERENT_CLASS_AGENTS
                        break
                        # encounter_different_class = True
                else:
                    self.map[-1, pos[0], pos[1], pos[2]] = m[pos[0], pos[1], pos[2]]
                    tmp_class_of_agent = ind
                    overlap = True
                    # return encounter_flags

    def step_each_else(self, agent, agent_ind):
        pos = agent.state.p_pos
        assert pos.dtype == int, str(pos.dtype)
        self.agents
        # Judge whether agent would walk out the scope.

        act = agent.action.u
        displayment = np.zeros(self.dim_p, dtype=np.int)
        if act == 1: displayment[2] = +1  # up
        if act == 2: displayment[2] = -1  # down
        if act == 3: displayment[0] = -1  # left
        if act == 4: displayment[0] = +1  # right
        if act == 5: displayment[1] = +1  # front
        if act == 6: displayment[1] = -1  # back
        next_pos = pos + displayment
        act = displayment

        if np.all(next_pos >= np.array([0, 0, 0])) and np.all(next_pos < self.shape):
            if (pos[2] == 0 and act[2] == 1) or \
                    ((pos[2] in [2, 4]) and (act[0] == 0)) or \
                    ((pos[2] in [1, 3]) and (act[1] == 0)):
                # agent.state.current_location_index += 1
                agent.state.p_pos = np.copy(next_pos)
                agent.state.track.append(agent.state.p_pos)
                # next_pos =
                # Check whether encounter others' tracks.
                # print('Before error', next_pos)
                # print(next_pos[0], next_pos[1], next_pos[2])
                # print(self.map[next_pos[0], next_pos[1], next_pos[2]])
                self.map[-1, next_pos[0], next_pos[1], next_pos[2]] = agent.class_index
                for ind, m in enumerate(self.map[:-1]):
                    if m[next_pos[0], next_pos[1], next_pos[2]] != 0:
                        if agent.class_index == self.agents[ind].class_index:  # Encounter agent in the same class
                            # self.map[-1, next_pos[0], next_pos[1], next_pos[2]] = m[next_pos[0], next_pos[1], next_pos[2]]
                            self.encounter_flags[agent_ind]['same'] = True
                        else:
                            self.map[-1, next_pos[0], next_pos[1], next_pos[2]] = ENCOUNTER_DIFFERENT_CLASS_AGENTS
                            self.encounter_flags[agent_ind]['diff'] = True

                            # self.map[agent.index, next_pos[0], next_pos[1], next_pos[2]] = agent.state.current_location_index

                            # return encounter_flags

    def done(self):
        num_classes = len(self.class_list)
        if np.max(label(self.map[-1])) == num_classes and np.max(self.map[-1]) == num_classes:
            return True
        return False

        # def reward(self):
        #     return 0

        # def observation(self, *args):
        #     return self.map
