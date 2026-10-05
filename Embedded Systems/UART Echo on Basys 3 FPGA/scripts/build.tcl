# vivado -mode batch -source scripts/build.tcl   (synth + impl + bitstream)
source [file join [file dirname [file normalize [info script]]] config.tcl]
open_project [file join $proj_dir $proj_name.xpr]

reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "Synthesis failed" }

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "Implementation failed" }

set bit [file join $proj_dir $proj_name.runs impl_1 ${top}.bit]
file mkdir [file join $root build]
file copy -force $bit [file join $root build ${top}.bit]
puts "Bitstream: [file join $root build ${top}.bit]"
open_run impl_1   ;# reports need the routed design open
report_timing_summary -file [file join $root build timing_summary.rpt]
report_utilization    -file [file join $root build utilization.rpt]
close_project
