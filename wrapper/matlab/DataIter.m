classdef DataIter
    properties %(Access = private)
        head_
        tail_
        handle_
    end

    methods
        function this = DataIter(cfg)
            assert(ischar(cfg));
            this.head_ = true;
            this.tail_ = false;
            this.handle_ = cxxnet_mex('MEXCXNIOCreateFromConfig', cfg);
        end
        function delete(this)
            cxxnet_mex('MEXCXNIOFree', this.handle_);
        end
        function ret = next(this)
            ret = cxxnet_mex('MEXCXNIONext', this.handle_);
            this.head_ = false;
            this.tail_ = ret == 0;
        end
        function before_first(this)
           cxxnet_mex('MEXCXNIOBeforeFirst', this.handle_);
           this.head_ = true;
           this.tail_ = false;
        end
        function check_valid(this)
            assert(this.head_ == true, 'iterator is at head');
            assert(this.tail_ == false, 'iterator is at end');
        end
        function data = get_data(this)
            if this.tail_ == false,
                data = cxxnet_mex('MEXCXNIOGetData', this.handle_);
            else
                printf('Iterator is at end\n');
            end
        end
        function label = get_label(this)
            if this.tail_ == false,
                label = cxxnet_mex('MEXCXNIOGetLabel', this.handle_);
            else
                printf('Iterator is at end\n');
            end
        end
    end
end
