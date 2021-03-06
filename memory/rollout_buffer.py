from memory.base_memory import BaseMemory
import numpy as np





#-------------------------------------------------------------------#
# Main memory class
#-------------------------------------------------------------------#
class RolloutBuffer(BaseMemory):
    def __init__(self, 
        buffer_size=10000, 
        variable_dict={'state':(1,)}
    ) -> None:
        super().__init__(buffer_size)
        self.variable = {}
        self.current_id = 0
        self.__count = 0
        if (len(variable_dict)==1):
            print(f'Create memory container with {len(variable_dict)} variable.')
        else:
            print(f'Create memory container with {len(variable_dict)} variables.')
        for var in variable_dict:
            self.variable[var] = np.zeros((self.buffer_size,) + \
                variable_dict[var], dtype=float)
            print(f'...Add \'{var}\' with shape {variable_dict[var]}...')

    def store(self, *args):
        if len(args) != len(self.variable):
            a = len(args)
            b = len(self.variable)
            print(f'Memory error: The number of variables ({a}) to be stored is different' +
                f' from number of variables in memory type ({b}).')
        else:
            var_i = 0
            added = True
            for var in self.variable:
                if np.shape(args[var_i]) != self.variable[var][self.current_id].shape:
                    print(f'Memory error: The shape of variable "{var}" is different' +
                    ' from the input shape.')
                    added = False
                    break
                else:
                    self.variable[var][self.current_id] = np.asarray(args[var_i])
                    var_i += 1
            if added:
                self.current_id += 1
                if self.current_id == self.buffer_size:
                    self.current_id = 0
                self.__count += 1
    
    def sample(self, sample_size=1):
        batch_size = min(min(sample_size, self.buffer_size), self.__count)
        if sample_size > self.__count:
            print('Memory warning: Number of samples is exceed number of records in memory.')
        batch_id = range(self.current_id - batch_size, self.current_id)
        batch = {}
        for var in self.variable:
            batch[var] = self.variable[var][batch_id]
        return batch
        





#-------------------------------------------------------------------#
# Main test
#-------------------------------------------------------------------#
if __name__ == '__main__':
    my_dict = {
        'state': (3,),
        'action': (1,)
    }
    buffer_size = 15
    my_buffer = RolloutBuffer(
        buffer_size=buffer_size,
        variable_dict=my_dict
    )
    print(my_buffer.variable['action'].shape)
    my_buffer.store([1,2,3],[1,2])
    my_buffer.store([2,3,4],[2])
    my_buffer.store([3,4,5],[3])
    my_buffer.store([4,5,6],[4])
    print(my_buffer.current_id)
    batch = my_buffer.sample(6)
    print(batch['state'])
    print(batch['action'])
    
