# Utils Function Documentation

The file `utils.R` contains several functions that will help to import data and create consistency between different outputs.  
It is possible to import and use the functions with the command:  
`source(path/to/utils.r)`  
The functions in the file are the following:

1. import_data(data)
2. create_data_numb(data)
3. create_data_sev(data)
4. NRMSE(pred, obs)
5. gini_plot(predicted_loss_cost, exposure)
6. gini_value(predicted_loss_cost, exposure)
7. lift_curve_plot(predicted_loss_cost, observed_loss_cost, n)

The first function, `import data` reads and loads the data into the global environment with all the column specifications.  
The other two functions: `create_data_numb` and `create_data_sev` are used to import the data and create the two splits(numbers and severity) for the xgboost package. You can ignore these. 
The function `NRMSE` returns the value of the NRMSE considering the vectors supplied in the arguments.  
The function `gini_plot` returns the plot of the Lorenz curve according to the vectors supplied in the arguments.  
The function `gini_value` returns the value of the gini index considering the vectors supplied in the arguments.  
The function `lift_curve_plot` returns the plot of the lift curves according the vectors supplied in the arguments. The `n` arguments takes the number of buckets the data will be divided into. It is sometime necessary, for better visualization, to transform the axis in logarithmic scale: `lift_curve_plot(predicted_loss_cost, observed_loss_cost, n) + scale_y_log10()`.  