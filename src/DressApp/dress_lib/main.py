import exif_cut
#segmかhaikeiどちらか一方だけimportする 両方importするとoutofmemoryエラーになる
#import segm
import haikei
from indexnet_matting.scripts import demo
from indexnet_matting.scripts import cut

import time

def image_cut():
    start = time.time()
    #exif_cut.exifcut_compression_risize("segm")#haikeiを実行するならコメントアウト
    #segm.cutting_out()#haikeiを実行するならコメントアウト
    exif_cut.exifcut_compression_risize("haikei")#segmを実行するならコメントアウト
    haikei.cutting_out()#segmを実行するならコメントアウト
    demo.infer()
    cut.cutting()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

image_cut()