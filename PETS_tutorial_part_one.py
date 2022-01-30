!pip install fastai -q --upgrade
from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
path = untar_data(URLs.PETS)
path.ls()
pets = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct = 0.2, seed= 100),
    get_y = using_attr(RegexLabeller(pat = r'^(.*)_\d+\..*$'), 'name'),
    item_tfms = Resize(460),
    batch_tfms = aug_transforms(size = 224, min_scale = 0.75)
)
dls = pets.dataloaders(path/'images')
dls.show_batch(max_n=4, figsize = (6,9))
