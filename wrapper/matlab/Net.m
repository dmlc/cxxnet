classdef Net
    properties
        handle_
    end

    methods
        function this = Net(dev, cfg)
            assert(ischar(dev));
            assert(ischar(cfg));
            this.handle_ = cxxnet_mex('MEXCXNNetCreate', dev, cfg);
        end
        function delete(this)
            assert(this.handle_ > 0);
            cxxnet_mex('MEXCXNNetFree', this.handle_);
        end
        function set_param(this, key, val)
            assert(ischar(key));
            assert(ischar(val));
            assert(this.handle_ > 0);
            cxxnet_mex('MEXCXNNetSetParam', this.handle_, key, val);
        end
        function init_model(this)
            assert(this.handle_ > 0);
            cxxnet_mex('MEXCXNNetInitModel', this.handle_);
        end
        function load_model(this, fname)
            assert(ischar(fname));
            assert(this.handle_ > 0);
            cxxnet_mex('MEXCXNNetLoadModel', this.handle_, fname);
        end
        function save_model(this, fname)
            assert(ischar(fname));
            assert(this.handle_ > 0);
            cxxnet_mex('MEXCXNNetSaveModel', this.handle_, fname);
        end
        function start_round(this, round_counter)
            assert(isnumeric(round_counter));
            assert(this.handle_ > 0);
            cxxnet_mex('MEXCXNNetStartRound', this.handle_, round_counter);
        end
        function update(this, data, label)
            assert(this.handle_ > 0);
            if isnumeric(data) == 0,
                assert(data.handle_ > 0);
                data.check_valid();
                cxxnet_mex('MEXCXNNetUpdateIter', this.handle_, data.handle_);
            else
               assert(isnumeric(data));
               assert(isnumeric(label));
               data = single(data);
               label = single(label);
               assert(ndims(data) == 4);
               sz1 = size(data);
               sz2 = size(label);
               if ndims(label) == 1,
                  label = reshape(label, sz2(1), 1);
               end
               assert(ndims(label) == 2);

               assert(sz1(1) == sz2(1));
               cxxnet_mex('MEXCXNNetUpdateBatch', this.handle_, data, label);
            end
        end
        function ret = evaluate(this, data, name)
            assert(this.handle_ > 0);
            assert(isnumeric(data) == 0, 'Only support data iter now');
            assert(data.handle_ > 0);
            assert(ischar(name));
            ret = cxxnet_mex('MEXCXNNetEvaluate', this.handle_, data.handle_, name);

        end
        function ret = predict(this, data)
            assert(this.handle_ > 0);
            if isnumeric(data) == 0,
                assert(data.handle_ > 0);
                data.check_valid();
                ret = cxxnet_mex('MEXCXNNetPredictIter', this.handle_, data.handle_);
            else
                assert(ndims(data) == 4);
                data = single(data);
                ret = cxxnet_mex('MEXCXNNetPredictBatch', this.handle_, data);
            end
        end
        function ret = extract(this, data, name)
            assert(this.handle_ > 0);
            if isnumeric(data) == 0,
                assert(data.handle_ > 0);
                data.check_valid();
                ret = cxxnet_mex('MEXCXNNetExtractIter', this.handle_, data.handle_, name);
            else
                assert(ndims(data) == 4);
                data = single(data);
                ret = cxxnet_mex('MEXCXNNetExtractBatch', this.handle_, data, name);
            end
        end
        function set_weight(this, weight, layer_name, tag)
            assert(this.handle_ > 0);
            assert(strcmp(tag, 'wmat') || strcmp(tag, 'bias'));
            assert(isnumeric(weight));
            assert(ischar(layer_name));
            weight = single(weight);
            cxxnet_mex('MEXCXNNetSetWeight', this.handle_, weight, layer_name, tag);
        end
        function ret = get_weight(this, layer_name, tag)
            assert(this.handle_ > 0);
            assert(strcmp(tag, 'wmat') || strcmp(tag, 'bias'));
            assert(ischar(layer_name));
            ret = cxxnet_mex('MEXCXNNetGetWeight', this.handle_, layer_name, tag);
        end
    end

end
