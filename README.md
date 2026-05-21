
## what I want to track

 - OoO lock aquire
 - track lock ownership transfer
 - total number of CS performed
 - percent of time with thread in CS
 - average per thread wait time

 ## to do

 - write csv module
 - write figures module
 - write runner script
 - init repo
 - verify on different machine
 - dependancies?
 - write README ig


 ## timeline
 
 finish writing project and have results  
 5-25


 ## results dir structure


/files
    /logs
        offsets.txt
        
    /csv
        /runs
            /run_id1 `parameters`
                /data
                    iter1
                    ...
                    iter10
                /op_timeline
                    iter1
                    ...
                    iter10
                avg_metrics1
                ...
                avg_metricsN
            /run_id2
            ...
        /aggragate
            `decide later what metrics are importaint` 
    /figures






    /batch
        /batch1
        /batch2
        ...
 