import logging

import numpy as np

logger = logging.getLogger(__name__)


#############################################
def update_dataset_possizes(main_loader, val_loader=None, seed=None, incr=0):
    """
    update_dataset_possizes(main_loader)
    """

    if not main_loader.dataset.same_possizes:
        return

    if seed is None:
        seed = np.random.choice(int(2**32))
    
    main_loader.dataset.set_possizes(seed=seed + incr, reset=True)
    
    if val_loader is not None:
        val_loader.dataset.set_possizes(seed=seed + incr, reset=True)    

    return seed


#############################################
def update_unexp(main_loader, val_loader=None, epoch_n=0, unexp_epoch=0):
    """
    update_unexp(main_loader)
    """

    if not main_loader.dataset.unexp and epoch_n >= unexp_epoch:
        main_loader.dataset.unexp = True
        loader_mode = main_loader.dataset.mode
        dataset_str = f" {loader_mode} dataset"
        if val_loader is not None and not val_loader.dataset.unexp:
            val_loader.dataset.unexp = True
            dataset_str = (
                f"{loader_mode} and {val_loader.dataset.mode} datasets"
                )

        logger.info(f"Setting {dataset_str} to include unexpected sequences.", 
            extra={"spacing": "\n"}
            )


#############################################
def update_gabors(main_loader, val_loader=None, seed=None, epoch_n=0, 
                  unexp_epoch=0):
    """
    update_gabors(main_loader)
    """

    seed = update_dataset_possizes(
        main_loader, val_loader, seed=seed, incr=epoch_n
        )
    
    update_unexp(
        main_loader, val_loader, epoch_n=epoch_n, 
        unexp_epoch=unexp_epoch
        )

    return seed

