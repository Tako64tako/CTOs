from . import exif_cut
from . import segm
from . import haikei
from .indexnet_matting.scripts import demo
from .indexnet_matting.scripts import cut

def image_cut():
    exif_cut.exifcut_compression_risize("haikei")
    segm.cutting_out()
    demo.infer()
    cut.cutting()