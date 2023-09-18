from optparse import Option, OptionParser

class ExtendedOption(Option):
    # Based on http://docs.python.org/library/optparse.html#adding-new-actions
    ACTIONS = Option.ACTIONS + ("extend",)
    STORE_ACTIONS = Option.STORE_ACTIONS + ("extend",)
    TYPED_ACTIONS = Option.TYPED_ACTIONS + ("extend",)
    ALWAYS_TYPED_ACTIONS = Option.ALWAYS_TYPED_ACTIONS + ("extend",)

    def take_action(self, action, dest, opt, value, values, parser):
        if action == "extend":
            lvalue = value.split(",")
            values.ensure_value(dest, []).extend(lvalue)
        else:
            Option.take_action(
                self, action, dest, opt, value, values, parser)


def parseOptions_nonsocial():
    optParser = OptionParser(option_class=ExtendedOption)
    optParser = add_nonsocial_options(optParser)

    opts, args = optParser.parse_args()

    return opts


def parseOptions_social():
    optParser = OptionParser(option_class=ExtendedOption)
    optParser = add_nonsocial_options(optParser)
    optParser.add_option('-d', '--demonstrators', action='store', type='int', dest='demonstrators',
                         default='1',
                         help='Number of random selected buildings that act as demonstrators (lower or equal than the '
                              'number of total buildings). If not defined, one building acts as demonstrator.')
    optParser.add_option('--sac', action='store_true', default=False, dest='exclude_sac',
                         help='Do not train a soft actor-critic (SAC) agent for comparison.')
    optParser.add_option('--sacdb2', action='store_true', default=False, dest='exclude_sacdb2',
                         help='Do not train a DB2 soft actor-critic (SAC) agent for comparison.')
    optParser.add_option('--mode', action='store', type='int', dest='mode', default='1',
                         help='Social-learning mode to use.')
    optParser.add_option('--ir', action='store', type='float', dest='ir', default='0.01',
                         help='Imitation rate for imitating actions/values of the demonstrators.')
    optParser.add_option('--pretrained_demonstrator', action='store', type='string', dest='pretrained_demonstrator',
                         default=None,
                         help='Path to pretained demonstrator agent to use. '
                              'Overwrites the number of demonstrators to use.')
    optParser.add_option('--transitions', action='store', type='string', dest='demo_transitions', default=None,
                         help='Path to transitions stored as csv from a demonstrator to put in the replay buffer.')

    opts, args = optParser.parse_args()

    return opts


def add_nonsocial_options(optParser):
    optParser.add_option('-s', '--schema', action='store', type='string', dest='schema',
                         default='nydata',
                         help='Name of the directory including the schema and data files')
    optParser.add_option('-r', '--randomseed', action='store', type='int', dest='seed',
                         help='Random seed to create reproducible results. If not defined, no fixed seed is used.')
    optParser.add_option('--batch', action='store', type='int', dest='batch',
                         help='Batch size.')
    optParser.add_option('--discount', action='store', type='float', dest='discount', default='0.99',
                         help='Discount factor')
    optParser.add_option('-b', '--buildings', action='store', type='int', dest='buildings',
                         help='Number of random selected buildings to include in training (between 1 and 15). '
                              'If not defined, all buildings are included.')
    optParser.add_option('-e', '--episodes', action='store', type='int', dest='episodes',
                         default='128',
                         help='Number of training epsiodes')
    optParser.add_option('--tql', action='store_true', default=False, dest='exclude_tql',
                         help='Do not train a Tabular Q-Learning agent for comparison.')
    optParser.add_option('--rbc', action='store_true', default=False, dest='exclude_rbc',
                         help='Do not train a rule-based control agent for comparison.')
    optParser.add_option('--autotune', action='store_true', default=False, dest='autotune',
                         help='Autotune the entropy value of the SAC agent.')
    optParser.add_option('--clipgradient', action='store_true', default=False, dest='clipgradient',
                         help='Clip gradients during training.')
    optParser.add_option('--kaiming', action='store_true', default=False, dest='kaiming',
                         help='Use kaiming initialization for the networks')
    optParser.add_option('--l2', action='store_true', default=False, dest='l2_loss',
                         help='Use l2 loss for critic networks')
    optParser.add_option('-o', '--observation', action='extend', type='string', dest='observations',
                         help='Comma separated list of observations that should be active. '
                              'If not defined, the full observation space (as defined in the schema file) is used.')
    optParser.add_option('--building_ids', action='append', type='int', dest='building_ids',
                         help='Ids of the buildings that should be trained. Overwrites the building_count.')
    optParser.add_option('--store', action='store_true', default=False, dest='store_agents',
                         help='Store trained agents.')

    return optParser

