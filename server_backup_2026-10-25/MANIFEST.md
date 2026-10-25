# 服务器原始数据备份（2026-10-25 21:36）

来源：10.10.52.210 /mnt/mydisk/xinguan-dendritic/bench（老盘）+ ~/bench（home）

## server_primary_backup.tar.gz（99 文件，546 KB）
- S1 T10 b0/b10 分片 ×16（10-24 21:27 原始轮）
- S1 T20 全部 c2t20_* ×24
- S2 原始分片：anchor_budget_gpu0/1（round-1）、s2r2_b2.part0/1（B2 轮）
- S2 r7/r8 系列 ×14（EWC/γ/M 网格）
- S1 b1/b3 原始轮 GPU 日志 ×16（分片被 E2 覆盖后，日志是逐臂汇总的旁证）
- c2t20 轮日志

## server_aux_logs.tar.gz（13 文件，44 KB）
- S4（s4ctx/s4nodend/s4diag）与 S3（cnnc2/s5bgn）的运行日志

## 说明
- S1 T10 b1/b3 的分片真值在 git 5c9d912（已恢复进本地 S1 目录），本包日志为第二旁证
- anchor_budget_all.json 服务器上无（本地 repo 内为修复后版本）
- 备份动因：服务器有自动重启/掉盘前科，此包落本地后达成三层冗余（服务器+本地+git）
