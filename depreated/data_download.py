from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode

ds_dict = MsDataset.load('person_detection_for_train', namespace="modelscope", split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD) # train set
# ds_dict = MsDataset.load('person_detection_for_train', namespace="modelscope", split='validation', download_mode=DownloadMode.FORCE_REDOWNLOAD) # val set

print(next(iter(ds_dict)))