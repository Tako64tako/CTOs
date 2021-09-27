# CTOs
## プロジェクト演習用レポジトリ
オンラインで試着サービスするシステムです.
機械学習で得た情報を元にあなたの体型や好み合わせた服をおすすめします.

## Requirements
| 言語/FW | Version|
| :------------| ---------: |
| python | 3.8.3　|

## Usage 
At the first execution
```sh
$ git clone https://github.com/Tako64tako/CTOs.git
$ cd CTOs
```

setup
```sh
$ docker-compose up -d --build
```
build success

http://localhost:1317/

finish
```sh
$ docker-compose down
```
if error
```sh
$ docker-compose down --remove-orphans
```
  
## Licence  
<a href="https://github.com/Tako64tako/CTOs/blob/main/LICENSE">GNU General Public License v3.0</a>

