# vivado -mode batch -source scripts/program.tcl   (Basys 3 must be connected, powered on)
source [file join [file dirname [file normalize [info script]]] config.tcl]
set bit [file join $root build ${top}.bit]
if {![file exists $bit]} { error "Missing $bit - run build.tcl first" }

open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices xc7a35t_0] 0]
current_hw_device $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
puts "Programmed $bit"
close_hw_manager
