# gcc升级

centos的缺点就是东西比较老，很多时候用一些软件会有莫名其妙的版本问题

我们装一个gcc7.5，至于其他版本的，方法都是一样的

**不需要root**

首先到[gnu](https://ftp.gnu.org/gnu/gcc/gcc-7.5.0/)网站上下载对应的gcc版本，我下载的是7.5版本

```shell
mkdir gcc&&cd gcc

wget https://ftp.gnu.org/gnu/gcc/gcc-7.5.0/gcc-7.5.0.tar.gz --no-check-certificate

tar -zxvf gcc-7.5.0.tar.gz

cd gcc-7.5.0
```



**下载gcc依赖，十分重要**

```shell
./contrib/download_prerequisites
```

建一个build文件夹

```shell
mkdir gcc-build-7.5.0

cd  cd gcc-build-7.5.0/
```

检查是否能够make

```shell
../configure -enable-checking=release -enable-languages=c,c++ -disable-multilib
```

make的时候一定要指定线程数，就搞32个吧，如果不指定，干5个小时也是有可能的

```shell
make -j32
```

指定位置安装，否则会安装到root用户下,我们建一个自己家目录下的文件夹

```shell
mkdir /beegfs/home/rlong/soft/gcc/gcc-install-7.5.0
make DESTDIR=/beegfs/home/rlong/soft/gcc/gcc-install-7.5.0 install
```

然后install

```shell
make DESTDIR=/beegfs/home/rlong/soft/gcc/gcc-install-7.5.0 install
```

这样gcc就安装成功了，修改一下环境变量就行了

```shell
vim ~/.bashrc
```

把刚才DESTDIR里面的路径添加一下，这里是我自己的路径,修改的时候最好先做好备份(.bashrc.bak)，出了问题再替换回来

```shell
export LD_LIBRARY_PATH=/beegfs/home/rlong/soft/gcc/gcc-install-7.5.0/usr/local/lib64:$LD_LIBRARY_PATH
export PATH=/beegfs/home/rlong/soft/gcc/gcc-install-7.5.0/usr/local/bin:$PATH
```

最后

```shell
source ~/.bashrc
```

查看gcc版本

```shell
gcc -v
```

