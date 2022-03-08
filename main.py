
""" Entry point for training
"""


from train_options import TrainOptions
from trainer_vln import TrainerVLN
from trainer_vln_no_map import TrainerVLN_UnknownMap
from tester import VLNTester
from tester_no_map import VLNTesterUnknownMap

import multiprocessing as mp
from multiprocessing import Pool, TimeoutError


def nav_testing(options, scene_id):
    if options.vln:
        tester = VLNTester(options, scene_id)
    elif options.vln_no_map:
        tester = VLNTesterUnknownMap(options, scene_id)
    tester.test_navigation()


if __name__ == '__main__':

    options = TrainOptions().parse_args()

    if options.is_train:

        if options.vln:
            trainer = TrainerVLN(options)
        elif options.vln_no_map:
            trainer = TrainerVLN_UnknownMap(options)
        trainer.train()
    else:

        scene_ids = options.scenes_list
        # Create iterables for map function
        n = len(scene_ids)
        options_list = [options] * n
        args = [*zip(options_list, scene_ids)]

        # isolate OpenGL context in each simulator instance
        with Pool(processes=options.gpu_capacity) as pool:
            pool.starmap(nav_testing, args)
    
    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
