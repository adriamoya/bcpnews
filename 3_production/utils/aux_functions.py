
def pause_execution():

    # Proceed ?
    while True:
        keep_ = input("Do you wish to proceed? [y/n]: ") # raw_input for python2
        if keep_.lower() in ["y", ""]:
            print("")
            break
        elif keep_.lower() == "n": # abort the process
            print("\nStopping process ...\n")
            exit()
        else:
            print("\nPlease type y or n.\n")
