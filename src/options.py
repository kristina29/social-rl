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


def parseOptions():
    optParser = OptionParser(option_class=ExtendedOption)
    optParser.add_option('-s', '--schema', action='store', type='string', dest='schema',
                         default='test',
                         help='Name of the directory including the schema and data files')
    optParser.add_option('-r', '--randomseed', action='store', type='int', dest='seed',
                         help='Random seed to create reproducible results. If not defined, no fixed seed is used.')
    optParser.add_option('-b', '--buildings', action='store', type='int', dest='buildings',
                         help='Number of random selected buildings to include in training (between 1 and 15). '
                              'If not defined, all buildings are included.')
    optParser.add_option('-d', '--days', action='store', type='int', dest='days',
                         help='Number of random selected consecutive days to include in training. '
                              'If not defined, all available days are used.')
    optParser.add_option('-e', '--episodes', action='store', type='int', dest='episodes',
                         default='128',
                         help='Number of training epsiodes')
    optParser.add_option('--tql', action='store_true', default=False, dest='exclude_tql',
                         help='Do not train a Tabular Q-Learning agent for comparison.')
    optParser.add_option('--rbc', action='store_true', default=False, dest='exclude_rbc',
                         help='Do not train a rule-based control agent for comparison.')
    optParser.add_option('-o', '--observation', action='extend', type='string', dest='observations',
                         help='Comma separated list of observations that should be active. '
                              'If not defined, the full observation space (as defined in the schema file) is used.')

    opts, args = optParser.parse_args()

    return opts
