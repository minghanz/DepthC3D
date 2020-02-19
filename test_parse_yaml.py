from options import MonodepthOptions
import yaml

options = MonodepthOptions()
opts, rest = options.parse()
if opts.config_file:
    with open(opts.config_file) as f:
        data = yaml.safe_load(f)
        # print(data)
        arg_dict = opts.__dict__
        # for key, value in data.items():
        #     print(key, ":", value)
        #     arg_dict[key] = value
        arg_dict.update(data)

print(opts.min_depth)
print(type(opts.min_depth))
