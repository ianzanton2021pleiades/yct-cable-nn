@echo off
cd /d "%~dp0"
call conda activate gpushare_cu124
python "%~dp0yct-TDR-2-GUI-multisamples-FreqSel_causal_fixed_nn.pyw"
pause
