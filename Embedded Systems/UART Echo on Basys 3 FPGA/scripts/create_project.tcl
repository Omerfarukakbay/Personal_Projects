# vivado -mode batch -source scripts/create_project.tcl
source [file join [file dirname [file normalize [info script]]] config.tcl]

create_project -force $proj_name $proj_dir -part $part
set_property target_language Verilog [current_project]

add_files -fileset sources_1 [glob -nocomplain $root/rtl/*.v $root/rtl/*.sv]
# [list ...] keeps the path as one item even though it contains a space ("Digital Design")
add_files -fileset constrs_1 [list [file join $root constraints basys3.xdc]]
add_files -fileset sim_1     [glob -nocomplain $root/sim/*.v $root/sim/*.sv]

set_property top $top        [get_filesets sources_1]
set_property top $sim_top    [get_filesets sim_1]
set_property -name xsim.simulate.runtime -value {-all} -objects [get_filesets sim_1]
update_compile_order -fileset sources_1
puts "Project created: $proj_dir/$proj_name.xpr"
close_project
