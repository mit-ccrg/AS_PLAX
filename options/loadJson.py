import argparse
import json


#loop through dict to get args
def get_argument(nested_dictionary):
    argument = []
    for key, value in nested_dictionary.items():
        if type(value) is dict:
            tmp = get_argument(value)
            for arg in tmp:
                argument.append(arg)
        else:
            argument.append('--' + str(key))
            argument.append(str(value))
    return argument

class LoadJson(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if '.json' in values:
            config = json.load(open(values))
            #only load needed options for training or testing
            if (config["basic_options"]["is_train"] == "testing"):
                config.pop('train_options', None)
            else:
                config.pop('test_options', None)

            argument = get_argument(config)   
            parser.parse_args(argument, namespace)
