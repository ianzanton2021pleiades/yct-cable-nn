# FDR Reader UI 处理链

日期：2026-11-10。

范围：消费者阅读版的静态托管代码与BAML审阅。重算调用进入厂商原生DLL；本文不推断DLL内部数值公式。

## 显示平滑

`Calculation.Movmean(double[,], K, column)`为每个输出点复制原距离值，并对指定响应列取居中均值，返回新数组。`K>1`时半窗宽为`floor(K/2)`，窗口在数组首尾收缩；所以K=2实际平均3点，K=15平均15点，K=75平均75点。K≤1时只复制响应值，不平滑。实现见`extracted/ui/Calculation.cs`第42–73行。

分析窗对脉冲和阶跃原响应调用此方法后再绘图；主窗口阶跃图及多测次比较图也直接传入响应原数组。调用见`extracted/ui/Window_Analysis.cs`第260–268、351–365行，以及`extracted/ui/MainWindow.cs`第485–486、710行。主视图平滑参数`MovmeanK`与对比视图参数`MovmeanK_Contrast`分别保存，见`MainWindow.cs`第41–43、1004–1045行。

本次焦点页复现同一窗口算法：每张厂商响应表先完整平滑，之后按审阅窗口裁剪并为图形降采样。页面列出脉冲与阶跃各自的距离采样步长，以及K=2、15、75对应的内部窗口跨度。该步骤不产生新响应表，也不从缺失表的S11补算曲线。

## 比较图Y偏移

阶跃比较图的偏移按钮把文本框数值传给当前`CurrentSweep`对应的绘图项，见`MainWindow.cs`第978–982行。`LineGraph.OffsetPointsY_Temp`调用`Illustration.OffsetPoints_Y`，将该项已生成的`ActualPointList`中每个`point.Y`加上常数后刷新图形；见`LineGraph.cs`第972–978行、`Illustration.cs`第155–159行。此操作不改写`StepResponseOrigin`或FDR表。

石杨对照页的Y对齐按100–180m阶跃均值对齐A相，每相仅加常数偏移；页面同时保留未对齐的原始绝对响应，并逐项列出K=2与K=75的偏移量。没有缩放或去趋势。

## 显式重新计算与写回

可确认的重算入口是BAML中的Ctrl+R命令`RoutedCommand_Recalculate`，没有对应菜单项或按钮文字。命令处理器清空当前测次脉冲/阶跃缓存并启动后台计算。扫测列表的`SelectedItem`经`Sweep_Show`设为`CurrentSweep`，因此重算目标是当前选中的单个测次，可以单独选中C相复测。见`extracted/ui/MainWindow.cs`第311–317、437–447、1142–1165行；BAML资源位置与调用路径见`后端响应处理链.md`。

`MainWindow.Recalculate`把该测次的`MeasurementData`及`MeasureSetting`参数传给`Analyser.Arithmetic.GetResponse(..., ref step, ref impulse, true)`。IL调用token将目标类型确认在嵌入程序集`Analyser`的`Analyser.Arithmetic`中；它不是同名的`FDR_Reader.Arithmetic`。`isMul=true`进入`Calculate_Impulse_Response_Process`。包装方法与DllImport见`extracted/ui/Analyser.Arithmetic.cs`第79、94行，调用点见`MainWindow.cs`第1163行。

计算结束后，程序按当前测次的文件路径与ID先删除旧响应表，再新建并写入`StepResponse_<id>`和`ImpulseResponse_<id>`。见`extracted/backend/SQL_Sweep_manual_write.cs`第3–24行和`MainWindow.cs`第1164–1165行。直接对原始FDR执行Ctrl+R会改写该文件。

本次只对原FDR做只读静态分析和数据读取，没有启动阅读版、执行Ctrl+R或写回任何源FDR。石杨rowid=3（C相初测）单独保留；邦润rowid=3的C相响应表缺失继续标缺失，rowid=4相位未知且不进入三相比较。完整审阅包保持原样，仅重新生成焦点页。

## 证据位置

- `extracted/ui/Calculation.cs`：MovingMean实现。
- `extracted/ui/Window_Analysis.cs`、`MainWindow.cs`：显示平滑与当前测次选择。
- `extracted/ui/LineGraph.cs`、`Illustration.cs`：比较图偏移只作用于绘图点。
- `extracted/ui/Analyser.Arithmetic.cs`：托管响应计算包装与原生函数绑定。
- `extracted/backend/SQL_Sweep_manual_write.cs`：FDR响应表写回。
- `后端响应处理链.md`：自动加载、Ctrl+R命令绑定与原生计算边界。
