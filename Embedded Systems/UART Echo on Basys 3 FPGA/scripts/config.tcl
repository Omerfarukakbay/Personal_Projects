# Shared settings. Edit TOP / SIM_TOP if you rename modules.
set script_dir [file dirname [file normalize [info script]]]
set root       [file normalize [file join $script_dir ..]]
set proj_name  uart_basys3
set proj_dir   [file join $root build $proj_name]
set part       xc7a35tcpg236-1
set top        top
set sim_top    tb_top
