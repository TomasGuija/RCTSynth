# dam_datamodule.py
from typing import Optional, Sequence
import h5py
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
import dam.src as dam

class H5VolumesDataset(Dataset):
    """
    Reuses your dam.generators.Volumes, but ensures each worker
    owns its own h5py.File handle (safer with num_workers>0).
    """
    def __init__(
        self,
        h5_path: str,
        indices: Sequence[int],
        *,
        masks: bool,
        xkey: str,
        ykey: str,
        xmask: str,
        ymask: str,
        maxv: float,
        minv: float,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.indices = list(indices)

        # open per-worker file handle and build the inner dataset
        self._fh = h5py.File(self.h5_path, "r")
        self._inner = dam.generators.Volumes(
            list_IDs=self.indices,
            file_handle=self._fh,
            masks=masks,
            xkey=xkey,
            ykey=ykey,
            xmask=xmask,
            ymask=ymask,
            maxv=maxv,
            minv=minv,
        )

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        # Returns exactly what your generator returns:
        #   (planning, repeat) or (planning, repeat, planning_mask, repeat_mask)
        return self._inner[idx]

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass


class DamDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        dataset_path: str,
        xkey: str,
        ykey: str,
        xmask: str,
        ymask: str,
        maxv: float,
        minv: float,
        train_split: float,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.xkey = xkey
        self.ykey = ykey
        self.xmask = xmask
        self.ymask = ymask
        self.maxv = maxv
        self.minv = minv
        self.train_split = train_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        # Discover N and inshape once (on main process)
        with h5py.File(self.dataset_path, "r") as fh:
            # data stored as [X, Y, Z, N]; your generator does np.transpose and adds channel
            N = fh[self.xkey].shape[-1]
            # inshape must match what DamBase expects (X,Y,Z)
            self.inshape = tuple(reversed(fh[self.xkey].shape[:-1]))  # keep same as your original

        # split
        all_ids = list(range(N))
        train_count = max(1, round(self.train_split * N))
        self.train_ids = all_ids[:train_count]
        self.val_ids = all_ids[train_count:] or [all_ids[-1]]

    def setup(self, stage: Optional[str] = None):
        # Always create fresh datasets (safe for DDP)
        self.train_set = H5VolumesDataset(
            h5_path=self.dataset_path,
            indices=self.train_ids,
            masks=True,
            xkey=self.xkey,
            ykey=self.ykey,
            xmask=self.xmask,
            ymask=self.ymask,
            maxv=self.maxv,
            minv=self.minv,
        )
        self.val_set = H5VolumesDataset(
            h5_path=self.dataset_path,
            indices=self.val_ids,
            masks=True,
            xkey=self.xkey,
            ykey=self.ykey,
            xmask=self.xmask,
            ymask=self.ymask,
            maxv=self.maxv,
            minv=self.minv,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
