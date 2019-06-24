<!--
generated using

dir="/Users/abduld/.gvm/pkgsets/go1.10/global/src/github.com/rai-project/tensorcore_bench/docs/figures/";
baseName[f_]:=
If[DirectoryName[f]===dir,
FileBaseName[f],
FileNameJoin[{FileBaseName[DirectoryName[f]],FileBaseName[f]}]
];
files=baseName/@FileNames["*png",dir,Infinity];
makeAnchor[f_]:=StringReplace[f,{ " "->"-"}];
tocTemplate=StringTemplate["- [`file`](#`anchor`)"];
figureTemplate=StringTemplate["## `file` \n\n [pdf](`file`.pdf), [png](`file`.png), [html](`file`.html),[view_html](https://tcuscope.netlify.com/figures/`file`.html)\n\n[![`file`](`file`.png)](`file`)"];
StringRiffle[{
"## Table Of Contents\n",
StringRiffle[TemplateApply[tocTemplate,<|"file"->#,"anchor"->makeAnchor[#]|>]&/@files,"\n"],
"\n\n\n----------------------------------------\n",
StringRiffle[TemplateApply[figureTemplate,<|"file"->#,"anchor"->makeAnchor[#]|>]&/@files,"\n\n\n----------------------------------------\n"]
},
"\n"
]//CopyToClipboard

-->


## Table Of Contents

- [full_prefixsum_atomic_cg](#full_prefixsum_atomic_cg)
- [full_prefixsum_ker3](#full_prefixsum_ker3)
- [full_prefixsum](#full_prefixsum)
- [full_reduction_atomic_ballot](#full_reduction_atomic_ballot)
- [full_reduction_atomic_sync](#full_reduction_atomic_sync)
- [full_reduction_cg](#full_reduction_cg)
- [full_reduction_cub](#full_reduction_cub)
- [full_reduction_ker2](#full_reduction_ker2)
- [full_reduction](#full_reduction)
- [full_reduction_thrust](#full_reduction_thrust)
- [full_reduction_unsafe](#full_reduction_unsafe)
- [gemm_16_16](#gemm_16_16)
- [gemm_16_32](#gemm_16_32)
- [gemm_large](#gemm_large)
- [gemm_medium](#gemm_medium)
- [gemm_small](#gemm_small)
- [gemv](#gemv)
- [hgemm](#hgemm)
- [memory_read_global_memory_coalesced](#memory_read_global_memory_coalesced)
- [memory_read_global_memory_unit](#memory_read_global_memory_unit)
- [memory_read_local_memory](#memory_read_local_memory)
- [memory_read_write_global](#memory_read_write_global)
- [memory_write_global_memory_coalesced](#memory_write_global_memory_coalesced)
- [memory_write_global_memory_unit](#memory_write_global_memory_unit)
- [memory_write_local_memory](#memory_write_local_memory)
- [segmented_prefixsum_16n_1024](#segmented_prefixsum_16n_1024)
- [segmented_prefixsum_16n_16](#segmented_prefixsum_16n_16)
- [segmented_prefixsum_16n_256](#segmented_prefixsum_16n_256)
- [segmented_prefixsum_16n_32](#segmented_prefixsum_16n_32)
- [segmented_prefixsum_16n_64](#segmented_prefixsum_16n_64)
- [segmented_prefixsum_16n_block](#segmented_prefixsum_16n_block)
- [segmented_prefixsum_16n_block_unsafe](#segmented_prefixsum_16n_block_unsafe)
- [segmented_prefixsum_16n_opt](#segmented_prefixsum_16n_opt)
- [segmented_prefixsum_16n_opt_unsafe](#segmented_prefixsum_16n_opt_unsafe)
- [segmented_prefixsum_16n](#segmented_prefixsum_16n)
- [segmented_prefixsum_16n_unsafe](#segmented_prefixsum_16n_unsafe)
- [segmented_prefixsum_16](#segmented_prefixsum_16)
- [segmented_prefixsum_16_unsafe](#segmented_prefixsum_16_unsafe)
- [segmented_prefixsum_256n_1024](#segmented_prefixsum_256n_1024)
- [segmented_prefixsum_256n_16384](#segmented_prefixsum_256n_16384)
- [segmented_prefixsum_256n_2048](#segmented_prefixsum_256n_2048)
- [segmented_prefixsum_256n_256](#segmented_prefixsum_256n_256)
- [segmented_prefixsum_256n_4096](#segmented_prefixsum_256n_4096)
- [segmented_prefixsum_256n_block](#segmented_prefixsum_256n_block)
- [segmented_prefixsum_256n_block_unsafe](#segmented_prefixsum_256n_block_unsafe)
- [segmented_prefixsum_256n_opt](#segmented_prefixsum_256n_opt)
- [segmented_prefixsum_256n](#segmented_prefixsum_256n)
- [segmented_prefixsum_256n_unsafe](#segmented_prefixsum_256n_unsafe)
- [segmented_prefixsum_256](#segmented_prefixsum_256)
- [segmented_prefixsum_256_unsafe](#segmented_prefixsum_256_unsafe)
- [segmented_prefixsum_cub_block](#segmented_prefixsum_cub_block)
- [segmented_prefixsum_cub_warp](#segmented_prefixsum_cub_warp)
- [segmented_prefixsum_thrust](#segmented_prefixsum_thrust)
- [segmented_prefixsum_thrust_tune](#segmented_prefixsum_thrust_tune)
- [segmented_reduction_16n_1024](#segmented_reduction_16n_1024)
- [segmented_reduction_16n_16](#segmented_reduction_16n_16)
- [segmented_reduction_16n_256](#segmented_reduction_16n_256)
- [segmented_reduction_16n_32](#segmented_reduction_16n_32)
- [segmented_reduction_16n_64](#segmented_reduction_16n_64)
- [segmented_reduction_16n_block](#segmented_reduction_16n_block)
- [segmented_reduction_16n_block_unsafe](#segmented_reduction_16n_block_unsafe)
- [segmented_reduction_16n_opt](#segmented_reduction_16n_opt)
- [segmented_reduction_16n_opt_unsafe](#segmented_reduction_16n_opt_unsafe)
- [segmented_reduction_16n](#segmented_reduction_16n)
- [segmented_reduction_16n_unsafe](#segmented_reduction_16n_unsafe)
- [segmented_reduction_16](#segmented_reduction_16)
- [segmented_reduction_16_unsafe](#segmented_reduction_16_unsafe)
- [segmented_reduction_256n_1024](#segmented_reduction_256n_1024)
- [segmented_reduction_256n_16384](#segmented_reduction_256n_16384)
- [segmented_reduction_256n_2048](#segmented_reduction_256n_2048)
- [segmented_reduction_256n_256](#segmented_reduction_256n_256)
- [segmented_reduction_256n_4096](#segmented_reduction_256n_4096)
- [segmented_reduction_256n_block](#segmented_reduction_256n_block)
- [segmented_reduction_256n_block_unsafe](#segmented_reduction_256n_block_unsafe)
- [segmented_reduction_256n](#segmented_reduction_256n)
- [segmented_reduction_256n_unsafe](#segmented_reduction_256n_unsafe)
- [segmented_reduction_256](#segmented_reduction_256)
- [segmented_reduction_256_unsafe](#segmented_reduction_256_unsafe)
- [segmented_reduction_cub_block](#segmented_reduction_cub_block)
- [segmented_reduction_cub_device](#segmented_reduction_cub_device)
- [segmented_reduction_cub_device_tune](#segmented_reduction_cub_device_tune)
- [segmented_reduction_cub_warp](#segmented_reduction_cub_warp)
- [segmented_reduction_thrust](#segmented_reduction_thrust)
- [tune_segmented_prefixsum_16n_all](#tune_segmented_prefixsum_16n_all)
- [tune_segmented_prefixsum_16n_block](#tune_segmented_prefixsum_16n_block)
- [tune_segmented_prefixsum_16n_block_unsafe](#tune_segmented_prefixsum_16n_block_unsafe)
- [tune_segmented_prefixsum_16n_opt](#tune_segmented_prefixsum_16n_opt)
- [tune_segmented_prefixsum_16n_opt_unsafe](#tune_segmented_prefixsum_16n_opt_unsafe)
- [tune_segmented_prefixsum_16n](#tune_segmented_prefixsum_16n)
- [tune_segmented_prefixsum_16n_unsafe](#tune_segmented_prefixsum_16n_unsafe)
- [tune_segmented_prefixsum_256n_all](#tune_segmented_prefixsum_256n_all)
- [tune_segmented_prefixsum_256n_block](#tune_segmented_prefixsum_256n_block)
- [tune_segmented_prefixsum_256n_block_unsafe](#tune_segmented_prefixsum_256n_block_unsafe)
- [tune_segmented_prefixsum_256n](#tune_segmented_prefixsum_256n)
- [tune_segmented_prefixsum_256n_unsafe](#tune_segmented_prefixsum_256n_unsafe)
- [tune_segmented_prefixsum](#tune_segmented_prefixsum)
- [tune_segmented_reduction_16n_all](#tune_segmented_reduction_16n_all)
- [tune_segmented_reduction_16n_block](#tune_segmented_reduction_16n_block)
- [tune_segmented_reduction_16n_block_unsafe](#tune_segmented_reduction_16n_block_unsafe)
- [tune_segmented_reduction_16n_opt](#tune_segmented_reduction_16n_opt)
- [tune_segmented_reduction_16n_opt_unsafe](#tune_segmented_reduction_16n_opt_unsafe)
- [tune_segmented_reduction_16n](#tune_segmented_reduction_16n)
- [tune_segmented_reduction_16n_unsafe](#tune_segmented_reduction_16n_unsafe)
- [tune_segmented_reduction_256n_all](#tune_segmented_reduction_256n_all)
- [tune_segmented_reduction_256n_block](#tune_segmented_reduction_256n_block)
- [tune_segmented_reduction_256n_block_unsafe](#tune_segmented_reduction_256n_block_unsafe)
- [tune_segmented_reduction_256n](#tune_segmented_reduction_256n)
- [tune_segmented_reduction_256n_unsafe](#tune_segmented_reduction_256n_unsafe)
- [tune_segmented_reduction_cub_device](#tune_segmented_reduction_cub_device)
- [tune_segmented_reduction](#tune_segmented_reduction)
- [tune_segmented_reduction_thrust](#tune_segmented_reduction_thrust)



----------------------------------------

## full_prefixsum_atomic_cg 

 [pdf](full_prefixsum_atomic_cg.pdf), [png](full_prefixsum_atomic_cg.png), [html](full_prefixsum_atomic_cg.html),[view_html](https://tcuscope.netlify.com/figures/full_prefixsum_atomic_cg.html)

[![full_prefixsum_atomic_cg](full_prefixsum_atomic_cg.png)](full_prefixsum_atomic_cg)


----------------------------------------
## full_prefixsum_ker3 

 [pdf](full_prefixsum_ker3.pdf), [png](full_prefixsum_ker3.png), [html](full_prefixsum_ker3.html),[view_html](https://tcuscope.netlify.com/figures/full_prefixsum_ker3.html)

[![full_prefixsum_ker3](full_prefixsum_ker3.png)](full_prefixsum_ker3)


----------------------------------------
## full_prefixsum 

 [pdf](full_prefixsum.pdf), [png](full_prefixsum.png), [html](full_prefixsum.html),[view_html](https://tcuscope.netlify.com/figures/full_prefixsum.html)

[![full_prefixsum](full_prefixsum.png)](full_prefixsum)


----------------------------------------
## full_reduction_atomic_ballot 

 [pdf](full_reduction_atomic_ballot.pdf), [png](full_reduction_atomic_ballot.png), [html](full_reduction_atomic_ballot.html),[view_html](https://tcuscope.netlify.com/figures/full_reduction_atomic_ballot.html)

[![full_reduction_atomic_ballot](full_reduction_atomic_ballot.png)](full_reduction_atomic_ballot)


----------------------------------------
## full_reduction_atomic_sync 

 [pdf](full_reduction_atomic_sync.pdf), [png](full_reduction_atomic_sync.png), [html](full_reduction_atomic_sync.html),[view_html](https://tcuscope.netlify.com/figures/full_reduction_atomic_sync.html)

[![full_reduction_atomic_sync](full_reduction_atomic_sync.png)](full_reduction_atomic_sync)


----------------------------------------
## full_reduction_cg 

 [pdf](full_reduction_cg.pdf), [png](full_reduction_cg.png), [html](full_reduction_cg.html),[view_html](https://tcuscope.netlify.com/figures/full_reduction_cg.html)

[![full_reduction_cg](full_reduction_cg.png)](full_reduction_cg)


----------------------------------------
## full_reduction_cub 

 [pdf](full_reduction_cub.pdf), [png](full_reduction_cub.png), [html](full_reduction_cub.html),[view_html](https://tcuscope.netlify.com/figures/full_reduction_cub.html)

[![full_reduction_cub](full_reduction_cub.png)](full_reduction_cub)


----------------------------------------
## full_reduction_ker2 

 [pdf](full_reduction_ker2.pdf), [png](full_reduction_ker2.png), [html](full_reduction_ker2.html),[view_html](https://tcuscope.netlify.com/figures/full_reduction_ker2.html)

[![full_reduction_ker2](full_reduction_ker2.png)](full_reduction_ker2)


----------------------------------------
## full_reduction 

 [pdf](full_reduction.pdf), [png](full_reduction.png), [html](full_reduction.html),[view_html](https://tcuscope.netlify.com/figures/full_reduction.html)

[![full_reduction](full_reduction.png)](full_reduction)


----------------------------------------
## full_reduction_thrust 

 [pdf](full_reduction_thrust.pdf), [png](full_reduction_thrust.png), [html](full_reduction_thrust.html),[view_html](https://tcuscope.netlify.com/figures/full_reduction_thrust.html)

[![full_reduction_thrust](full_reduction_thrust.png)](full_reduction_thrust)


----------------------------------------
## full_reduction_unsafe 

 [pdf](full_reduction_unsafe.pdf), [png](full_reduction_unsafe.png), [html](full_reduction_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/full_reduction_unsafe.html)

[![full_reduction_unsafe](full_reduction_unsafe.png)](full_reduction_unsafe)


----------------------------------------
## gemm_16_16 

 [pdf](gemm_16_16.pdf), [png](gemm_16_16.png), [html](gemm_16_16.html),[view_html](https://tcuscope.netlify.com/figures/gemm_16_16.html)

[![gemm_16_16](gemm_16_16.png)](gemm_16_16)


----------------------------------------
## gemm_16_32 

 [pdf](gemm_16_32.pdf), [png](gemm_16_32.png), [html](gemm_16_32.html),[view_html](https://tcuscope.netlify.com/figures/gemm_16_32.html)

[![gemm_16_32](gemm_16_32.png)](gemm_16_32)


----------------------------------------
## gemm_large 

 [pdf](gemm_large.pdf), [png](gemm_large.png), [html](gemm_large.html),[view_html](https://tcuscope.netlify.com/figures/gemm_large.html)

[![gemm_large](gemm_large.png)](gemm_large)


----------------------------------------
## gemm_medium 

 [pdf](gemm_medium.pdf), [png](gemm_medium.png), [html](gemm_medium.html),[view_html](https://tcuscope.netlify.com/figures/gemm_medium.html)

[![gemm_medium](gemm_medium.png)](gemm_medium)


----------------------------------------
## gemm_small 

 [pdf](gemm_small.pdf), [png](gemm_small.png), [html](gemm_small.html),[view_html](https://tcuscope.netlify.com/figures/gemm_small.html)

[![gemm_small](gemm_small.png)](gemm_small)


----------------------------------------
## gemv 

 [pdf](gemv.pdf), [png](gemv.png), [html](gemv.html),[view_html](https://tcuscope.netlify.com/figures/gemv.html)

[![gemv](gemv.png)](gemv)


----------------------------------------
## hgemm 

 [pdf](hgemm.pdf), [png](hgemm.png), [html](hgemm.html),[view_html](https://tcuscope.netlify.com/figures/hgemm.html)

[![hgemm](hgemm.png)](hgemm)


----------------------------------------
## memory_read_global_memory_coalesced 

 [pdf](memory_read_global_memory_coalesced.pdf), [png](memory_read_global_memory_coalesced.png), [html](memory_read_global_memory_coalesced.html),[view_html](https://tcuscope.netlify.com/figures/memory_read_global_memory_coalesced.html)

[![memory_read_global_memory_coalesced](memory_read_global_memory_coalesced.png)](memory_read_global_memory_coalesced)


----------------------------------------
## memory_read_global_memory_unit 

 [pdf](memory_read_global_memory_unit.pdf), [png](memory_read_global_memory_unit.png), [html](memory_read_global_memory_unit.html),[view_html](https://tcuscope.netlify.com/figures/memory_read_global_memory_unit.html)

[![memory_read_global_memory_unit](memory_read_global_memory_unit.png)](memory_read_global_memory_unit)


----------------------------------------
## memory_read_local_memory 

 [pdf](memory_read_local_memory.pdf), [png](memory_read_local_memory.png), [html](memory_read_local_memory.html),[view_html](https://tcuscope.netlify.com/figures/memory_read_local_memory.html)

[![memory_read_local_memory](memory_read_local_memory.png)](memory_read_local_memory)


----------------------------------------
## memory_read_write_global 

 [pdf](memory_read_write_global.pdf), [png](memory_read_write_global.png), [html](memory_read_write_global.html),[view_html](https://tcuscope.netlify.com/figures/memory_read_write_global.html)

[![memory_read_write_global](memory_read_write_global.png)](memory_read_write_global)


----------------------------------------
## memory_write_global_memory_coalesced 

 [pdf](memory_write_global_memory_coalesced.pdf), [png](memory_write_global_memory_coalesced.png), [html](memory_write_global_memory_coalesced.html),[view_html](https://tcuscope.netlify.com/figures/memory_write_global_memory_coalesced.html)

[![memory_write_global_memory_coalesced](memory_write_global_memory_coalesced.png)](memory_write_global_memory_coalesced)


----------------------------------------
## memory_write_global_memory_unit 

 [pdf](memory_write_global_memory_unit.pdf), [png](memory_write_global_memory_unit.png), [html](memory_write_global_memory_unit.html),[view_html](https://tcuscope.netlify.com/figures/memory_write_global_memory_unit.html)

[![memory_write_global_memory_unit](memory_write_global_memory_unit.png)](memory_write_global_memory_unit)


----------------------------------------
## memory_write_local_memory 

 [pdf](memory_write_local_memory.pdf), [png](memory_write_local_memory.png), [html](memory_write_local_memory.html),[view_html](https://tcuscope.netlify.com/figures/memory_write_local_memory.html)

[![memory_write_local_memory](memory_write_local_memory.png)](memory_write_local_memory)


----------------------------------------
## segmented_prefixsum_16n_1024 

 [pdf](segmented_prefixsum_16n_1024.pdf), [png](segmented_prefixsum_16n_1024.png), [html](segmented_prefixsum_16n_1024.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_1024.html)

[![segmented_prefixsum_16n_1024](segmented_prefixsum_16n_1024.png)](segmented_prefixsum_16n_1024)


----------------------------------------
## segmented_prefixsum_16n_16 

 [pdf](segmented_prefixsum_16n_16.pdf), [png](segmented_prefixsum_16n_16.png), [html](segmented_prefixsum_16n_16.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_16.html)

[![segmented_prefixsum_16n_16](segmented_prefixsum_16n_16.png)](segmented_prefixsum_16n_16)


----------------------------------------
## segmented_prefixsum_16n_256 

 [pdf](segmented_prefixsum_16n_256.pdf), [png](segmented_prefixsum_16n_256.png), [html](segmented_prefixsum_16n_256.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_256.html)

[![segmented_prefixsum_16n_256](segmented_prefixsum_16n_256.png)](segmented_prefixsum_16n_256)


----------------------------------------
## segmented_prefixsum_16n_32 

 [pdf](segmented_prefixsum_16n_32.pdf), [png](segmented_prefixsum_16n_32.png), [html](segmented_prefixsum_16n_32.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_32.html)

[![segmented_prefixsum_16n_32](segmented_prefixsum_16n_32.png)](segmented_prefixsum_16n_32)


----------------------------------------
## segmented_prefixsum_16n_64 

 [pdf](segmented_prefixsum_16n_64.pdf), [png](segmented_prefixsum_16n_64.png), [html](segmented_prefixsum_16n_64.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_64.html)

[![segmented_prefixsum_16n_64](segmented_prefixsum_16n_64.png)](segmented_prefixsum_16n_64)


----------------------------------------
## segmented_prefixsum_16n_block 

 [pdf](segmented_prefixsum_16n_block.pdf), [png](segmented_prefixsum_16n_block.png), [html](segmented_prefixsum_16n_block.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_block.html)

[![segmented_prefixsum_16n_block](segmented_prefixsum_16n_block.png)](segmented_prefixsum_16n_block)


----------------------------------------
## segmented_prefixsum_16n_block_unsafe 

 [pdf](segmented_prefixsum_16n_block_unsafe.pdf), [png](segmented_prefixsum_16n_block_unsafe.png), [html](segmented_prefixsum_16n_block_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_block_unsafe.html)

[![segmented_prefixsum_16n_block_unsafe](segmented_prefixsum_16n_block_unsafe.png)](segmented_prefixsum_16n_block_unsafe)


----------------------------------------
## segmented_prefixsum_16n_opt 

 [pdf](segmented_prefixsum_16n_opt.pdf), [png](segmented_prefixsum_16n_opt.png), [html](segmented_prefixsum_16n_opt.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_opt.html)

[![segmented_prefixsum_16n_opt](segmented_prefixsum_16n_opt.png)](segmented_prefixsum_16n_opt)


----------------------------------------
## segmented_prefixsum_16n_opt_unsafe 

 [pdf](segmented_prefixsum_16n_opt_unsafe.pdf), [png](segmented_prefixsum_16n_opt_unsafe.png), [html](segmented_prefixsum_16n_opt_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_opt_unsafe.html)

[![segmented_prefixsum_16n_opt_unsafe](segmented_prefixsum_16n_opt_unsafe.png)](segmented_prefixsum_16n_opt_unsafe)


----------------------------------------
## segmented_prefixsum_16n 

 [pdf](segmented_prefixsum_16n.pdf), [png](segmented_prefixsum_16n.png), [html](segmented_prefixsum_16n.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n.html)

[![segmented_prefixsum_16n](segmented_prefixsum_16n.png)](segmented_prefixsum_16n)


----------------------------------------
## segmented_prefixsum_16n_unsafe 

 [pdf](segmented_prefixsum_16n_unsafe.pdf), [png](segmented_prefixsum_16n_unsafe.png), [html](segmented_prefixsum_16n_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16n_unsafe.html)

[![segmented_prefixsum_16n_unsafe](segmented_prefixsum_16n_unsafe.png)](segmented_prefixsum_16n_unsafe)


----------------------------------------
## segmented_prefixsum_16 

 [pdf](segmented_prefixsum_16.pdf), [png](segmented_prefixsum_16.png), [html](segmented_prefixsum_16.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16.html)

[![segmented_prefixsum_16](segmented_prefixsum_16.png)](segmented_prefixsum_16)


----------------------------------------
## segmented_prefixsum_16_unsafe 

 [pdf](segmented_prefixsum_16_unsafe.pdf), [png](segmented_prefixsum_16_unsafe.png), [html](segmented_prefixsum_16_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_16_unsafe.html)

[![segmented_prefixsum_16_unsafe](segmented_prefixsum_16_unsafe.png)](segmented_prefixsum_16_unsafe)


----------------------------------------
## segmented_prefixsum_256n_1024 

 [pdf](segmented_prefixsum_256n_1024.pdf), [png](segmented_prefixsum_256n_1024.png), [html](segmented_prefixsum_256n_1024.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_1024.html)

[![segmented_prefixsum_256n_1024](segmented_prefixsum_256n_1024.png)](segmented_prefixsum_256n_1024)


----------------------------------------
## segmented_prefixsum_256n_16384 

 [pdf](segmented_prefixsum_256n_16384.pdf), [png](segmented_prefixsum_256n_16384.png), [html](segmented_prefixsum_256n_16384.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_16384.html)

[![segmented_prefixsum_256n_16384](segmented_prefixsum_256n_16384.png)](segmented_prefixsum_256n_16384)


----------------------------------------
## segmented_prefixsum_256n_2048 

 [pdf](segmented_prefixsum_256n_2048.pdf), [png](segmented_prefixsum_256n_2048.png), [html](segmented_prefixsum_256n_2048.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_2048.html)

[![segmented_prefixsum_256n_2048](segmented_prefixsum_256n_2048.png)](segmented_prefixsum_256n_2048)


----------------------------------------
## segmented_prefixsum_256n_256 

 [pdf](segmented_prefixsum_256n_256.pdf), [png](segmented_prefixsum_256n_256.png), [html](segmented_prefixsum_256n_256.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_256.html)

[![segmented_prefixsum_256n_256](segmented_prefixsum_256n_256.png)](segmented_prefixsum_256n_256)


----------------------------------------
## segmented_prefixsum_256n_4096 

 [pdf](segmented_prefixsum_256n_4096.pdf), [png](segmented_prefixsum_256n_4096.png), [html](segmented_prefixsum_256n_4096.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_4096.html)

[![segmented_prefixsum_256n_4096](segmented_prefixsum_256n_4096.png)](segmented_prefixsum_256n_4096)


----------------------------------------
## segmented_prefixsum_256n_block 

 [pdf](segmented_prefixsum_256n_block.pdf), [png](segmented_prefixsum_256n_block.png), [html](segmented_prefixsum_256n_block.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_block.html)

[![segmented_prefixsum_256n_block](segmented_prefixsum_256n_block.png)](segmented_prefixsum_256n_block)


----------------------------------------
## segmented_prefixsum_256n_block_unsafe 

 [pdf](segmented_prefixsum_256n_block_unsafe.pdf), [png](segmented_prefixsum_256n_block_unsafe.png), [html](segmented_prefixsum_256n_block_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_block_unsafe.html)

[![segmented_prefixsum_256n_block_unsafe](segmented_prefixsum_256n_block_unsafe.png)](segmented_prefixsum_256n_block_unsafe)


----------------------------------------
## segmented_prefixsum_256n_opt 

 [pdf](segmented_prefixsum_256n_opt.pdf), [png](segmented_prefixsum_256n_opt.png), [html](segmented_prefixsum_256n_opt.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_opt.html)

[![segmented_prefixsum_256n_opt](segmented_prefixsum_256n_opt.png)](segmented_prefixsum_256n_opt)


----------------------------------------
## segmented_prefixsum_256n 

 [pdf](segmented_prefixsum_256n.pdf), [png](segmented_prefixsum_256n.png), [html](segmented_prefixsum_256n.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n.html)

[![segmented_prefixsum_256n](segmented_prefixsum_256n.png)](segmented_prefixsum_256n)


----------------------------------------
## segmented_prefixsum_256n_unsafe 

 [pdf](segmented_prefixsum_256n_unsafe.pdf), [png](segmented_prefixsum_256n_unsafe.png), [html](segmented_prefixsum_256n_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256n_unsafe.html)

[![segmented_prefixsum_256n_unsafe](segmented_prefixsum_256n_unsafe.png)](segmented_prefixsum_256n_unsafe)


----------------------------------------
## segmented_prefixsum_256 

 [pdf](segmented_prefixsum_256.pdf), [png](segmented_prefixsum_256.png), [html](segmented_prefixsum_256.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256.html)

[![segmented_prefixsum_256](segmented_prefixsum_256.png)](segmented_prefixsum_256)


----------------------------------------
## segmented_prefixsum_256_unsafe 

 [pdf](segmented_prefixsum_256_unsafe.pdf), [png](segmented_prefixsum_256_unsafe.png), [html](segmented_prefixsum_256_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_256_unsafe.html)

[![segmented_prefixsum_256_unsafe](segmented_prefixsum_256_unsafe.png)](segmented_prefixsum_256_unsafe)


----------------------------------------
## segmented_prefixsum_cub_block 

 [pdf](segmented_prefixsum_cub_block.pdf), [png](segmented_prefixsum_cub_block.png), [html](segmented_prefixsum_cub_block.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_cub_block.html)

[![segmented_prefixsum_cub_block](segmented_prefixsum_cub_block.png)](segmented_prefixsum_cub_block)


----------------------------------------
## segmented_prefixsum_cub_warp 

 [pdf](segmented_prefixsum_cub_warp.pdf), [png](segmented_prefixsum_cub_warp.png), [html](segmented_prefixsum_cub_warp.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_cub_warp.html)

[![segmented_prefixsum_cub_warp](segmented_prefixsum_cub_warp.png)](segmented_prefixsum_cub_warp)


----------------------------------------
## segmented_prefixsum_thrust 

 [pdf](segmented_prefixsum_thrust.pdf), [png](segmented_prefixsum_thrust.png), [html](segmented_prefixsum_thrust.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_thrust.html)

[![segmented_prefixsum_thrust](segmented_prefixsum_thrust.png)](segmented_prefixsum_thrust)


----------------------------------------
## segmented_prefixsum_thrust_tune 

 [pdf](segmented_prefixsum_thrust_tune.pdf), [png](segmented_prefixsum_thrust_tune.png), [html](segmented_prefixsum_thrust_tune.html),[view_html](https://tcuscope.netlify.com/figures/segmented_prefixsum_thrust_tune.html)

[![segmented_prefixsum_thrust_tune](segmented_prefixsum_thrust_tune.png)](segmented_prefixsum_thrust_tune)


----------------------------------------
## segmented_reduction_16n_1024 

 [pdf](segmented_reduction_16n_1024.pdf), [png](segmented_reduction_16n_1024.png), [html](segmented_reduction_16n_1024.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_1024.html)

[![segmented_reduction_16n_1024](segmented_reduction_16n_1024.png)](segmented_reduction_16n_1024)


----------------------------------------
## segmented_reduction_16n_16 

 [pdf](segmented_reduction_16n_16.pdf), [png](segmented_reduction_16n_16.png), [html](segmented_reduction_16n_16.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_16.html)

[![segmented_reduction_16n_16](segmented_reduction_16n_16.png)](segmented_reduction_16n_16)


----------------------------------------
## segmented_reduction_16n_256 

 [pdf](segmented_reduction_16n_256.pdf), [png](segmented_reduction_16n_256.png), [html](segmented_reduction_16n_256.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_256.html)

[![segmented_reduction_16n_256](segmented_reduction_16n_256.png)](segmented_reduction_16n_256)


----------------------------------------
## segmented_reduction_16n_32 

 [pdf](segmented_reduction_16n_32.pdf), [png](segmented_reduction_16n_32.png), [html](segmented_reduction_16n_32.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_32.html)

[![segmented_reduction_16n_32](segmented_reduction_16n_32.png)](segmented_reduction_16n_32)


----------------------------------------
## segmented_reduction_16n_64 

 [pdf](segmented_reduction_16n_64.pdf), [png](segmented_reduction_16n_64.png), [html](segmented_reduction_16n_64.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_64.html)

[![segmented_reduction_16n_64](segmented_reduction_16n_64.png)](segmented_reduction_16n_64)


----------------------------------------
## segmented_reduction_16n_block 

 [pdf](segmented_reduction_16n_block.pdf), [png](segmented_reduction_16n_block.png), [html](segmented_reduction_16n_block.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_block.html)

[![segmented_reduction_16n_block](segmented_reduction_16n_block.png)](segmented_reduction_16n_block)


----------------------------------------
## segmented_reduction_16n_block_unsafe 

 [pdf](segmented_reduction_16n_block_unsafe.pdf), [png](segmented_reduction_16n_block_unsafe.png), [html](segmented_reduction_16n_block_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_block_unsafe.html)

[![segmented_reduction_16n_block_unsafe](segmented_reduction_16n_block_unsafe.png)](segmented_reduction_16n_block_unsafe)


----------------------------------------
## segmented_reduction_16n_opt 

 [pdf](segmented_reduction_16n_opt.pdf), [png](segmented_reduction_16n_opt.png), [html](segmented_reduction_16n_opt.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_opt.html)

[![segmented_reduction_16n_opt](segmented_reduction_16n_opt.png)](segmented_reduction_16n_opt)


----------------------------------------
## segmented_reduction_16n_opt_unsafe 

 [pdf](segmented_reduction_16n_opt_unsafe.pdf), [png](segmented_reduction_16n_opt_unsafe.png), [html](segmented_reduction_16n_opt_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_opt_unsafe.html)

[![segmented_reduction_16n_opt_unsafe](segmented_reduction_16n_opt_unsafe.png)](segmented_reduction_16n_opt_unsafe)


----------------------------------------
## segmented_reduction_16n 

 [pdf](segmented_reduction_16n.pdf), [png](segmented_reduction_16n.png), [html](segmented_reduction_16n.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n.html)

[![segmented_reduction_16n](segmented_reduction_16n.png)](segmented_reduction_16n)


----------------------------------------
## segmented_reduction_16n_unsafe 

 [pdf](segmented_reduction_16n_unsafe.pdf), [png](segmented_reduction_16n_unsafe.png), [html](segmented_reduction_16n_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16n_unsafe.html)

[![segmented_reduction_16n_unsafe](segmented_reduction_16n_unsafe.png)](segmented_reduction_16n_unsafe)


----------------------------------------
## segmented_reduction_16 

 [pdf](segmented_reduction_16.pdf), [png](segmented_reduction_16.png), [html](segmented_reduction_16.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16.html)

[![segmented_reduction_16](segmented_reduction_16.png)](segmented_reduction_16)


----------------------------------------
## segmented_reduction_16_unsafe 

 [pdf](segmented_reduction_16_unsafe.pdf), [png](segmented_reduction_16_unsafe.png), [html](segmented_reduction_16_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_16_unsafe.html)

[![segmented_reduction_16_unsafe](segmented_reduction_16_unsafe.png)](segmented_reduction_16_unsafe)


----------------------------------------
## segmented_reduction_256n_1024 

 [pdf](segmented_reduction_256n_1024.pdf), [png](segmented_reduction_256n_1024.png), [html](segmented_reduction_256n_1024.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n_1024.html)

[![segmented_reduction_256n_1024](segmented_reduction_256n_1024.png)](segmented_reduction_256n_1024)


----------------------------------------
## segmented_reduction_256n_16384 

 [pdf](segmented_reduction_256n_16384.pdf), [png](segmented_reduction_256n_16384.png), [html](segmented_reduction_256n_16384.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n_16384.html)

[![segmented_reduction_256n_16384](segmented_reduction_256n_16384.png)](segmented_reduction_256n_16384)


----------------------------------------
## segmented_reduction_256n_2048 

 [pdf](segmented_reduction_256n_2048.pdf), [png](segmented_reduction_256n_2048.png), [html](segmented_reduction_256n_2048.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n_2048.html)

[![segmented_reduction_256n_2048](segmented_reduction_256n_2048.png)](segmented_reduction_256n_2048)


----------------------------------------
## segmented_reduction_256n_256 

 [pdf](segmented_reduction_256n_256.pdf), [png](segmented_reduction_256n_256.png), [html](segmented_reduction_256n_256.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n_256.html)

[![segmented_reduction_256n_256](segmented_reduction_256n_256.png)](segmented_reduction_256n_256)


----------------------------------------
## segmented_reduction_256n_4096 

 [pdf](segmented_reduction_256n_4096.pdf), [png](segmented_reduction_256n_4096.png), [html](segmented_reduction_256n_4096.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n_4096.html)

[![segmented_reduction_256n_4096](segmented_reduction_256n_4096.png)](segmented_reduction_256n_4096)


----------------------------------------
## segmented_reduction_256n_block 

 [pdf](segmented_reduction_256n_block.pdf), [png](segmented_reduction_256n_block.png), [html](segmented_reduction_256n_block.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n_block.html)

[![segmented_reduction_256n_block](segmented_reduction_256n_block.png)](segmented_reduction_256n_block)


----------------------------------------
## segmented_reduction_256n_block_unsafe 

 [pdf](segmented_reduction_256n_block_unsafe.pdf), [png](segmented_reduction_256n_block_unsafe.png), [html](segmented_reduction_256n_block_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n_block_unsafe.html)

[![segmented_reduction_256n_block_unsafe](segmented_reduction_256n_block_unsafe.png)](segmented_reduction_256n_block_unsafe)


----------------------------------------
## segmented_reduction_256n 

 [pdf](segmented_reduction_256n.pdf), [png](segmented_reduction_256n.png), [html](segmented_reduction_256n.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n.html)

[![segmented_reduction_256n](segmented_reduction_256n.png)](segmented_reduction_256n)


----------------------------------------
## segmented_reduction_256n_unsafe 

 [pdf](segmented_reduction_256n_unsafe.pdf), [png](segmented_reduction_256n_unsafe.png), [html](segmented_reduction_256n_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256n_unsafe.html)

[![segmented_reduction_256n_unsafe](segmented_reduction_256n_unsafe.png)](segmented_reduction_256n_unsafe)


----------------------------------------
## segmented_reduction_256 

 [pdf](segmented_reduction_256.pdf), [png](segmented_reduction_256.png), [html](segmented_reduction_256.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256.html)

[![segmented_reduction_256](segmented_reduction_256.png)](segmented_reduction_256)


----------------------------------------
## segmented_reduction_256_unsafe 

 [pdf](segmented_reduction_256_unsafe.pdf), [png](segmented_reduction_256_unsafe.png), [html](segmented_reduction_256_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_256_unsafe.html)

[![segmented_reduction_256_unsafe](segmented_reduction_256_unsafe.png)](segmented_reduction_256_unsafe)


----------------------------------------
## segmented_reduction_cub_block 

 [pdf](segmented_reduction_cub_block.pdf), [png](segmented_reduction_cub_block.png), [html](segmented_reduction_cub_block.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_cub_block.html)

[![segmented_reduction_cub_block](segmented_reduction_cub_block.png)](segmented_reduction_cub_block)


----------------------------------------
## segmented_reduction_cub_device 

 [pdf](segmented_reduction_cub_device.pdf), [png](segmented_reduction_cub_device.png), [html](segmented_reduction_cub_device.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_cub_device.html)

[![segmented_reduction_cub_device](segmented_reduction_cub_device.png)](segmented_reduction_cub_device)


----------------------------------------
## segmented_reduction_cub_device_tune 

 [pdf](segmented_reduction_cub_device_tune.pdf), [png](segmented_reduction_cub_device_tune.png), [html](segmented_reduction_cub_device_tune.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_cub_device_tune.html)

[![segmented_reduction_cub_device_tune](segmented_reduction_cub_device_tune.png)](segmented_reduction_cub_device_tune)


----------------------------------------
## segmented_reduction_cub_warp 

 [pdf](segmented_reduction_cub_warp.pdf), [png](segmented_reduction_cub_warp.png), [html](segmented_reduction_cub_warp.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_cub_warp.html)

[![segmented_reduction_cub_warp](segmented_reduction_cub_warp.png)](segmented_reduction_cub_warp)


----------------------------------------
## segmented_reduction_thrust 

 [pdf](segmented_reduction_thrust.pdf), [png](segmented_reduction_thrust.png), [html](segmented_reduction_thrust.html),[view_html](https://tcuscope.netlify.com/figures/segmented_reduction_thrust.html)

[![segmented_reduction_thrust](segmented_reduction_thrust.png)](segmented_reduction_thrust)


----------------------------------------
## tune_segmented_prefixsum_16n_all 

 [pdf](tune_segmented_prefixsum_16n_all.pdf), [png](tune_segmented_prefixsum_16n_all.png), [html](tune_segmented_prefixsum_16n_all.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_16n_all.html)

[![tune_segmented_prefixsum_16n_all](tune_segmented_prefixsum_16n_all.png)](tune_segmented_prefixsum_16n_all)


----------------------------------------
## tune_segmented_prefixsum_16n_block 

 [pdf](tune_segmented_prefixsum_16n_block.pdf), [png](tune_segmented_prefixsum_16n_block.png), [html](tune_segmented_prefixsum_16n_block.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_16n_block.html)

[![tune_segmented_prefixsum_16n_block](tune_segmented_prefixsum_16n_block.png)](tune_segmented_prefixsum_16n_block)


----------------------------------------
## tune_segmented_prefixsum_16n_block_unsafe 

 [pdf](tune_segmented_prefixsum_16n_block_unsafe.pdf), [png](tune_segmented_prefixsum_16n_block_unsafe.png), [html](tune_segmented_prefixsum_16n_block_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_16n_block_unsafe.html)

[![tune_segmented_prefixsum_16n_block_unsafe](tune_segmented_prefixsum_16n_block_unsafe.png)](tune_segmented_prefixsum_16n_block_unsafe)


----------------------------------------
## tune_segmented_prefixsum_16n_opt 

 [pdf](tune_segmented_prefixsum_16n_opt.pdf), [png](tune_segmented_prefixsum_16n_opt.png), [html](tune_segmented_prefixsum_16n_opt.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_16n_opt.html)

[![tune_segmented_prefixsum_16n_opt](tune_segmented_prefixsum_16n_opt.png)](tune_segmented_prefixsum_16n_opt)


----------------------------------------
## tune_segmented_prefixsum_16n_opt_unsafe 

 [pdf](tune_segmented_prefixsum_16n_opt_unsafe.pdf), [png](tune_segmented_prefixsum_16n_opt_unsafe.png), [html](tune_segmented_prefixsum_16n_opt_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_16n_opt_unsafe.html)

[![tune_segmented_prefixsum_16n_opt_unsafe](tune_segmented_prefixsum_16n_opt_unsafe.png)](tune_segmented_prefixsum_16n_opt_unsafe)


----------------------------------------
## tune_segmented_prefixsum_16n 

 [pdf](tune_segmented_prefixsum_16n.pdf), [png](tune_segmented_prefixsum_16n.png), [html](tune_segmented_prefixsum_16n.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_16n.html)

[![tune_segmented_prefixsum_16n](tune_segmented_prefixsum_16n.png)](tune_segmented_prefixsum_16n)


----------------------------------------
## tune_segmented_prefixsum_16n_unsafe 

 [pdf](tune_segmented_prefixsum_16n_unsafe.pdf), [png](tune_segmented_prefixsum_16n_unsafe.png), [html](tune_segmented_prefixsum_16n_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_16n_unsafe.html)

[![tune_segmented_prefixsum_16n_unsafe](tune_segmented_prefixsum_16n_unsafe.png)](tune_segmented_prefixsum_16n_unsafe)


----------------------------------------
## tune_segmented_prefixsum_256n_all 

 [pdf](tune_segmented_prefixsum_256n_all.pdf), [png](tune_segmented_prefixsum_256n_all.png), [html](tune_segmented_prefixsum_256n_all.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_256n_all.html)

[![tune_segmented_prefixsum_256n_all](tune_segmented_prefixsum_256n_all.png)](tune_segmented_prefixsum_256n_all)


----------------------------------------
## tune_segmented_prefixsum_256n_block 

 [pdf](tune_segmented_prefixsum_256n_block.pdf), [png](tune_segmented_prefixsum_256n_block.png), [html](tune_segmented_prefixsum_256n_block.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_256n_block.html)

[![tune_segmented_prefixsum_256n_block](tune_segmented_prefixsum_256n_block.png)](tune_segmented_prefixsum_256n_block)


----------------------------------------
## tune_segmented_prefixsum_256n_block_unsafe 

 [pdf](tune_segmented_prefixsum_256n_block_unsafe.pdf), [png](tune_segmented_prefixsum_256n_block_unsafe.png), [html](tune_segmented_prefixsum_256n_block_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_256n_block_unsafe.html)

[![tune_segmented_prefixsum_256n_block_unsafe](tune_segmented_prefixsum_256n_block_unsafe.png)](tune_segmented_prefixsum_256n_block_unsafe)


----------------------------------------
## tune_segmented_prefixsum_256n 

 [pdf](tune_segmented_prefixsum_256n.pdf), [png](tune_segmented_prefixsum_256n.png), [html](tune_segmented_prefixsum_256n.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_256n.html)

[![tune_segmented_prefixsum_256n](tune_segmented_prefixsum_256n.png)](tune_segmented_prefixsum_256n)


----------------------------------------
## tune_segmented_prefixsum_256n_unsafe 

 [pdf](tune_segmented_prefixsum_256n_unsafe.pdf), [png](tune_segmented_prefixsum_256n_unsafe.png), [html](tune_segmented_prefixsum_256n_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum_256n_unsafe.html)

[![tune_segmented_prefixsum_256n_unsafe](tune_segmented_prefixsum_256n_unsafe.png)](tune_segmented_prefixsum_256n_unsafe)


----------------------------------------
## tune_segmented_prefixsum 

 [pdf](tune_segmented_prefixsum.pdf), [png](tune_segmented_prefixsum.png), [html](tune_segmented_prefixsum.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_prefixsum.html)

[![tune_segmented_prefixsum](tune_segmented_prefixsum.png)](tune_segmented_prefixsum)


----------------------------------------
## tune_segmented_reduction_16n_all 

 [pdf](tune_segmented_reduction_16n_all.pdf), [png](tune_segmented_reduction_16n_all.png), [html](tune_segmented_reduction_16n_all.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_16n_all.html)

[![tune_segmented_reduction_16n_all](tune_segmented_reduction_16n_all.png)](tune_segmented_reduction_16n_all)


----------------------------------------
## tune_segmented_reduction_16n_block 

 [pdf](tune_segmented_reduction_16n_block.pdf), [png](tune_segmented_reduction_16n_block.png), [html](tune_segmented_reduction_16n_block.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_16n_block.html)

[![tune_segmented_reduction_16n_block](tune_segmented_reduction_16n_block.png)](tune_segmented_reduction_16n_block)


----------------------------------------
## tune_segmented_reduction_16n_block_unsafe 

 [pdf](tune_segmented_reduction_16n_block_unsafe.pdf), [png](tune_segmented_reduction_16n_block_unsafe.png), [html](tune_segmented_reduction_16n_block_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_16n_block_unsafe.html)

[![tune_segmented_reduction_16n_block_unsafe](tune_segmented_reduction_16n_block_unsafe.png)](tune_segmented_reduction_16n_block_unsafe)


----------------------------------------
## tune_segmented_reduction_16n_opt 

 [pdf](tune_segmented_reduction_16n_opt.pdf), [png](tune_segmented_reduction_16n_opt.png), [html](tune_segmented_reduction_16n_opt.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_16n_opt.html)

[![tune_segmented_reduction_16n_opt](tune_segmented_reduction_16n_opt.png)](tune_segmented_reduction_16n_opt)


----------------------------------------
## tune_segmented_reduction_16n_opt_unsafe 

 [pdf](tune_segmented_reduction_16n_opt_unsafe.pdf), [png](tune_segmented_reduction_16n_opt_unsafe.png), [html](tune_segmented_reduction_16n_opt_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_16n_opt_unsafe.html)

[![tune_segmented_reduction_16n_opt_unsafe](tune_segmented_reduction_16n_opt_unsafe.png)](tune_segmented_reduction_16n_opt_unsafe)


----------------------------------------
## tune_segmented_reduction_16n 

 [pdf](tune_segmented_reduction_16n.pdf), [png](tune_segmented_reduction_16n.png), [html](tune_segmented_reduction_16n.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_16n.html)

[![tune_segmented_reduction_16n](tune_segmented_reduction_16n.png)](tune_segmented_reduction_16n)


----------------------------------------
## tune_segmented_reduction_16n_unsafe 

 [pdf](tune_segmented_reduction_16n_unsafe.pdf), [png](tune_segmented_reduction_16n_unsafe.png), [html](tune_segmented_reduction_16n_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_16n_unsafe.html)

[![tune_segmented_reduction_16n_unsafe](tune_segmented_reduction_16n_unsafe.png)](tune_segmented_reduction_16n_unsafe)


----------------------------------------
## tune_segmented_reduction_256n_all 

 [pdf](tune_segmented_reduction_256n_all.pdf), [png](tune_segmented_reduction_256n_all.png), [html](tune_segmented_reduction_256n_all.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_256n_all.html)

[![tune_segmented_reduction_256n_all](tune_segmented_reduction_256n_all.png)](tune_segmented_reduction_256n_all)


----------------------------------------
## tune_segmented_reduction_256n_block 

 [pdf](tune_segmented_reduction_256n_block.pdf), [png](tune_segmented_reduction_256n_block.png), [html](tune_segmented_reduction_256n_block.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_256n_block.html)

[![tune_segmented_reduction_256n_block](tune_segmented_reduction_256n_block.png)](tune_segmented_reduction_256n_block)


----------------------------------------
## tune_segmented_reduction_256n_block_unsafe 

 [pdf](tune_segmented_reduction_256n_block_unsafe.pdf), [png](tune_segmented_reduction_256n_block_unsafe.png), [html](tune_segmented_reduction_256n_block_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_256n_block_unsafe.html)

[![tune_segmented_reduction_256n_block_unsafe](tune_segmented_reduction_256n_block_unsafe.png)](tune_segmented_reduction_256n_block_unsafe)


----------------------------------------
## tune_segmented_reduction_256n 

 [pdf](tune_segmented_reduction_256n.pdf), [png](tune_segmented_reduction_256n.png), [html](tune_segmented_reduction_256n.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_256n.html)

[![tune_segmented_reduction_256n](tune_segmented_reduction_256n.png)](tune_segmented_reduction_256n)


----------------------------------------
## tune_segmented_reduction_256n_unsafe 

 [pdf](tune_segmented_reduction_256n_unsafe.pdf), [png](tune_segmented_reduction_256n_unsafe.png), [html](tune_segmented_reduction_256n_unsafe.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_256n_unsafe.html)

[![tune_segmented_reduction_256n_unsafe](tune_segmented_reduction_256n_unsafe.png)](tune_segmented_reduction_256n_unsafe)


----------------------------------------
## tune_segmented_reduction_cub_device 

 [pdf](tune_segmented_reduction_cub_device.pdf), [png](tune_segmented_reduction_cub_device.png), [html](tune_segmented_reduction_cub_device.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_cub_device.html)

[![tune_segmented_reduction_cub_device](tune_segmented_reduction_cub_device.png)](tune_segmented_reduction_cub_device)


----------------------------------------
## tune_segmented_reduction 

 [pdf](tune_segmented_reduction.pdf), [png](tune_segmented_reduction.png), [html](tune_segmented_reduction.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction.html)

[![tune_segmented_reduction](tune_segmented_reduction.png)](tune_segmented_reduction)


----------------------------------------
## tune_segmented_reduction_thrust 

 [pdf](tune_segmented_reduction_thrust.pdf), [png](tune_segmented_reduction_thrust.png), [html](tune_segmented_reduction_thrust.html),[view_html](https://tcuscope.netlify.com/figures/tune_segmented_reduction_thrust.html)

[![tune_segmented_reduction_thrust](tune_segmented_reduction_thrust.png)](tune_segmented_reduction_thrust)
