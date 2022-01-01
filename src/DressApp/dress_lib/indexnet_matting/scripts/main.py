#from . import demo#パッケージからimportするとき（このプログラムからを実行していないとき）
#from . import cut
import demo#パッケージからimportするとき（このプログラムからを実行するとき）
import cut

demo.infer()
cut.cutting()

#pythonのimportでのrootディレクトリは実行ファイルのある場所である。