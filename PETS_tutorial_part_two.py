"""
!pip install fastai -q --upgrade
from fastai.basics import *
from fastai.vision.all import *
from fastai.callback.all import *
path = untar_data(URLs.PETS)
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
"""

model = cnn_learner(dls, resnet34, metrics = error_rate)
model.fit_one_cycle(4)
model.unfreeze()
model.lr_find()
model.fit_one_cycle(4, lr_max = slice (6e-5,2e-4)) #replace numbers with range the your valley in your lr_find plot
model.save("perfecto")
model.recorder.plot_loss()
results = ClassificationInterpretation.from_learner(model)
results.plot_confusion_matrix(figsize=(12,12), dpi=60)
results.plot_top_losses(9, figsize = (16, 16))
