# vivado -mode batch -source scripts/sim.tcl   (behavioural sim, waveform in build/)
source [file join [file dirname [file normalize [info script]]] config.tcl]
open_project [file join $proj_dir $proj_name.xpr]
# launch_simulation already runs until $finish (create_project.tcl sets the
# sim runtime to -all). A second "run all" after $finish would restart the
# free-running clock and never stop, so there is none here.
launch_simulation -mode behavioral
close_sim
close_project
