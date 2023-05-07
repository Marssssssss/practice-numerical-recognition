具体的各种尝试结论都放在代码注释中。

- `run.bat` - 运行学习过程，可以自己编辑源码更换各种函数查看效果，学习后会生成图片并能够在控制台进行交互查看学习效果
- `image_reader.bat` - 查看测试集和样本集的图片

另外，git 上传大文件要用 `git-lfs`，不支持直传，因此数据文件需要自己下载，运行 data 下的 download_data.bat 即可。

`pip` 依赖库： `pytorch`, `pandas`, `matplotlib`

如果要用 `cuda` 加速，记得安装 `cuda` 和 `CuCNN`。
