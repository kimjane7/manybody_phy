from plot import sidebyside_predictions

def main():

    units_list = [[2**i] for i in range(3, 12)]
    sidebyside_predictions(units_list, 'layers1')
    
    #fixed_total_units = 128
    #units_list = [[i, fixed_total_units-i] for i in range(8, fixed_total_units, 8)]
    #sidebyside_predictions(units_list, 'layers2')
    
    
    #units_list = [[30,30,120], [30,60,90], [30,90,60], [30,120,30], [60,30,90], \
    #              [60,60,60], [60,90,30], [90,30,60], [90,60,30], [120,30,30]]
    #sidebyside_predictions(units_list, 'layers3')
    
    



if __name__ == "__main__":
    main()
