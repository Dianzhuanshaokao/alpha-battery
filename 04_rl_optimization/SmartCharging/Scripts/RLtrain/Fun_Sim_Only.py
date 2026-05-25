# For GEM-2 NC paper
import random, os
import pybamm as pb;import numpy as np;
pb.set_logging_level("ERROR")
from scipy.io import savemat;from pybamm import constants,exp,sqrt;
from multiprocessing import Queue, Process
from queue import Empty
import time, signal

###################################################################
#############    From Patrick from Github        ##################
###################################################################
class TimeoutFunc(object):
    """
    Lightweight timeout implementation using signals (Scheme 2).
    Does NOT spawn sub-processes, avoiding competition in parallel envs.
    Note: Only works on Unix-based systems.
    """
    def __init__(self, func, timeout=None, timeout_val=None):
        assert callable(func), 'Positional argument 1 must be a callable method'
        self.func = func
        self.timeout = timeout
        self.timeout_val = timeout_val

    def _handle_timeout(self, signum, frame):
        raise TimeoutError("Simulation exceeded time limit.")

    def __call__(self, *args, **kwargs):
        # Register the signal handler for SIGALRM
        old_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.timeout)  # Schedule the alarm
        
        try:
            Result_List = self.func(*args, **kwargs)
        except TimeoutError:
            Call_ref = RioCallback()
            Result_List = [
                self.timeout_val,
                self.timeout_val,
                Call_ref,
                self.timeout_val
            ]
        except Exception as e:
            # Re-raise other exceptions to let the environment handle them
            raise e
        finally:
            # Disable the alarm and restore the old handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
        return Result_List
###################################################################
#############    New functions from P3 - 221113        ############
###################################################################
# DEFINE my callback:
class RioCallback(pb.callbacks.Callback):
    def __init__(self, logfile=None):
        self.logfile = logfile
        self.success  = True
        if logfile is None:
            # Use pybamm's logger, which prints to command line
            self.logger = pb.logger
        else:
            # Use a custom logger, this will have its own level so set it to the same
            # level as the pybamm logger (users can override this)
            self.logger = pb.get_new_logger(__name__, logfile)
            self.logger.setLevel(pb.logger.level)
    
    def on_experiment_error(self, logs):
        self.success  = False
    def on_experiment_infeasible(self, logs):
        self.success  = False

class Experiment_error_infeasible(ValueError):
    pass


# define to kill too long runs - Jianbo Huang kindly writes this
class TimeoutError(Exception):
	pass
def handle_signal(signal_num, frame):
	raise TimeoutError


# Define a function to calculate concentration change, whether electrolyte being squeezed out or added in
def Cal_new_con_Update(Sol,Para):   # subscript r means the reservoir
    # Note: c_EC_r is the initial EC  concentraiton in the reservoir; 
    #       c_e_r  is the initial Li+ concentraiton in the reservoir;
    #############################################################################################################################  
    ###############################           Step-1 Prepare parameters:        #################################################
    #############################################################################################################################  
    L_p   =Para["Positive electrode thickness [m]"]
    L_n   =Para["Negative electrode thickness [m]"]
    L_s   =Para["Separator thickness [m]"]
    L_y   =Para["Electrode width [m]"]   # Also update to change A_cc and therefore Q
    L_z   =Para["Electrode height [m]"]
    c_EC_r_old=Para["Current solvent concentration in the reservoir [mol.m-3]"]   # add two new parameters for paper 2
    c_e_r_old =Para["Current electrolyte concentration in the reservoir [mol.m-3]"]
    # Initial EC concentration in JR
    c_EC_JR_old =Para["Bulk solvent concentration [mol.m-3]"]  # Be careful when multiplying with pore volume to get amount in mole. Because of electrolyte dry out, that will give more amount than real.   
    # LLI due to electrode,Ratio of EC and lithium is 1:1 -> EC amount consumed is LLINegSEI[-1]
    LLINegSEI = (
        Sol["Loss of lithium to SEI [mol]"].entries[-1] 
        - Sol["Loss of lithium to SEI [mol]"].entries[0] )
    LLINegSEIcr = (
        Sol["Loss of lithium to SEI on cracks [mol]"].entries[-1]
        - 
        Sol["Loss of lithium to SEI on cracks [mol]"].entries[0]
        )
    LLINegDeadLiP = (
        Sol["Loss of lithium to dead lithium plating [mol]"].entries[-1] 
        - Sol["Loss of lithium to dead lithium plating [mol]"].entries[0])
    LLINegLiP = (
        Sol["Loss of lithium to lithium plating [mol]"].entries[-1] 
        - Sol["Loss of lithium to lithium plating [mol]"].entries[0])
    cLi_Xavg  = Sol["X-averaged electrolyte concentration [mol.m-3]"].entries[-1] 
    # Pore volume change with time:
    PoreVolNeg_0 = Sol["X-averaged negative electrode porosity"].entries[0]*L_n*L_y*L_z;
    PoreVolSep_0 = Sol["X-averaged separator porosity"].entries[0]*L_s*L_y*L_z;
    PoreVolPos_0 = Sol["X-averaged positive electrode porosity"].entries[0]*L_p*L_y*L_z;
    PoreVolNeg_1 = Sol["X-averaged negative electrode porosity"].entries[-1]*L_n*L_y*L_z;
    PoreVolSep_1 = Sol["X-averaged separator porosity"].entries[-1]*L_s*L_y*L_z;
    PoreVolPos_1 = Sol["X-averaged positive electrode porosity"].entries[-1]*L_p*L_y*L_z;
    #############################################################################################################################  
    ##### Step-2 Determine How much electrolyte is added (from the reservoir to JR) or squeezed out (from JR to reservoir) ######
    #######################   and finish electrolyte mixing     #################################################################  
    Vol_Elely_Tot_old = Para["Current total electrolyte volume in whole cell [m3]"] 
    Vol_Elely_JR_old  = Para["Current total electrolyte volume in jelly roll [m3]"] 
    if Vol_Elely_Tot_old - Vol_Elely_JR_old < 0:
        print('Model error! Electrolyte in JR is larger than in the cell!')
    Vol_Pore_tot_old  = PoreVolNeg_0 + PoreVolSep_0 + PoreVolPos_0    # pore volume at start time of the run
    Vol_Pore_tot_new  = PoreVolNeg_1 + PoreVolSep_1 + PoreVolPos_1    # pore volume at end   time of the run, intrinsic variable 
    Vol_Pore_decrease = Vol_Elely_JR_old  - Vol_Pore_tot_new # WHY Vol_Elely_JR_old not Vol_Pore_tot_old here? Because even the last state of the last solution (n-1) and the first state of the next solution (n) can be slightly different! 
    # EC:lithium:SEI=2:2:1     for SEI=(CH2OCO2Li)2, but because of too many changes are required, change to 2:1:1 for now
    # update 230703: Simon insist we should do: EC:lithium:SEI  =  2:2:1 and change V_SEI accordingly 
    # Because inner and outer SEI partial molar volume is the same, just set one for whole SEI
    VmolSEI   = Para["Outer SEI partial molar volume [m3.mol-1]"] # 9.8e-5,
    VmolLiP   = Para["Lithium metal partial molar volume [m3.mol-1]"] # 1.3e-05
    VmolEC    = Para["EC partial molar volume [m3.mol-1]"]
    #################   KEY EQUATION FOR DRY-OUT MODEL                   #################
    # UPDATE 230525: assume the formation of dead lithium doesn’t consumed EC
    Vol_EC_consumed  =  ( LLINegSEI + LLINegSEIcr   ) * 1 * VmolEC    # Mark: either with 2 or not, will decide how fast electrolyte being consumed!
    Vol_Elely_need   = Vol_EC_consumed - Vol_Pore_decrease
    Vol_SEILiP_increase = 0.5*(
        (LLINegSEI+LLINegSEIcr) * VmolSEI 
        #+ LLINegLiP * VmolLiP
        )    #  volume increase due to SEI+total LiP 
    #################   KEY EQUATION FOR DRY-OUT MODEL                   #################

    Test_V = Vol_SEILiP_increase - Vol_Pore_decrease  #  This value should always be zero, but now not, which becomes the source of error!
    Test_V2= (Vol_Pore_tot_old - Vol_Elely_JR_old) / Vol_Elely_JR_old * 100; # Porosity errors due to first time step
    
    # Start from here, there are serveral variables to be determined:
    #   1) Vol_Elely_add, or Vol_Elely_squeezed; depends on conditions. easier for 'squeezed' condition
    #   2) Vol_Elely_Tot_new, should always equals to Vol_Elely_Tot_old -  Vol_EC_consumed;
    #   3) Vol_Elely_JR_new, for 'add' condition: see old code; for 'squeezed' condition, equals to pore volume in JR
    #   4) Ratio_Dryout, for 'add' condition: see old code; for 'squeezed' condition, equals to Vol_Elely_Tot_new/Vol_Pore_tot_new 
    #   5) Ratio_CeEC_JR and Ratio_CeLi_JR: for 'add' condition: see old code; for 'squeezed' condition, equals to 1 (unchanged)
    #   6) c_e_r_new and c_EC_r_new: for 'add' condition: equals to old ones (unchanged); for 'squeezed' condition, need to carefully calculate     
    #   7) Width_new: for 'add' condition: see old code; for 'squeezed' condition, equals to L_y (unchanged)
    Vol_Elely_Tot_new = Vol_Elely_Tot_old - Vol_EC_consumed;

    
    if Vol_Elely_need < 0:
        print('Electrolyte is being squeezed out, check plated lithium (active and dead)')
        Vol_Elely_squeezed = - Vol_Elely_need;   # Make Vol_Elely_squeezed>0 for simplicity 
        Vol_Elely_add = 0.0;
        Vol_Elely_JR_new = Vol_Pore_tot_new; 
        Ratio_Dryout = Vol_Elely_Tot_new / Vol_Elely_JR_new
        Ratio_CeEC_JR = 1.0;    # the concentrations in JR don't need to change
        Ratio_CeLi_JR = 1.0; 
        SqueezedLiMol   =  Vol_Elely_squeezed*cLi_Xavg;
        SqueezedECMol   =  Vol_Elely_squeezed*c_EC_JR_old;
        Vol_Elely_reservoir_old = Vol_Elely_Tot_old - Vol_Elely_JR_old; 
        LiMol_reservoir_new = Vol_Elely_reservoir_old*c_e_r_old  + SqueezedLiMol;
        ECMol_reservoir_new = Vol_Elely_reservoir_old*c_EC_r_old + SqueezedECMol;
        c_e_r_new= LiMol_reservoir_new / (Vol_Elely_reservoir_old+Vol_Elely_squeezed);
        c_EC_r_new= ECMol_reservoir_new / (Vol_Elely_reservoir_old+Vol_Elely_squeezed);
        Width_new = L_y; 
    else:   # this means Vol_Elely_need >= 0, therefore the folliwing script should works for Vol_Elely_need=0 as well!
        # Important: Calculate added electrolyte based on excessive electrolyte, can be: 1) added as required; 2) added some, but less than required; 3) added 0 
        Vol_Elely_squeezed = 0;  
        if Vol_Elely_Tot_old > Vol_Elely_JR_old:                             # This means Vol_Elely_JR_old = Vol_Pore_tot_old ()
            if Vol_Elely_Tot_old-Vol_Elely_JR_old >= Vol_Elely_need:         # 1) added as required
                Vol_Elely_add     = Vol_Elely_need;  
                Vol_Elely_JR_new  = Vol_Pore_tot_new;  # also equals to 'Vol_Pore_tot_new', or Vol_Pore_tot_old - Vol_Pore_decrease
                Ratio_Dryout = 1.0;
            else:                                                            # 2) added some, but less than required;                                                         
                Vol_Elely_add     = Vol_Elely_Tot_old - Vol_Elely_JR_old;   
                Vol_Elely_JR_new  = Vol_Elely_Tot_new;                       # This means Vol_Elely_JR_new <= Vol_Pore_tot_new
                Ratio_Dryout = Vol_Elely_JR_new/Vol_Pore_tot_new;
        else:                                                                # 3) added 0 
            Vol_Elely_add = 0;
            Vol_Elely_JR_new  = Vol_Elely_Tot_new; 
            Ratio_Dryout = Vol_Elely_JR_new/Vol_Pore_tot_new;

        # Next: start mix electrolyte based on previous defined equation
        # Lithium amount in liquid phase, at initial time point
        TotLi_Elely_JR_Old = Sol["Total lithium in electrolyte [mol]"].entries[-1]; # remember this is only in JR
        # Added lithium and EC amount in the added lithium electrolyte: - 
        AddLiMol   =  Vol_Elely_add*c_e_r_old
        AddECMol   =  Vol_Elely_add*c_EC_r_old
        # Total amount of Li and EC in current electrolyte:
        # Remember Li has two parts, initial and added; EC has three parts, initial, consumed and added
        TotLi_Elely_JR_New   = TotLi_Elely_JR_Old + AddLiMol
        TotECMol_JR   = Vol_Elely_JR_old*c_EC_JR_old - LLINegSEI + AddECMol  # EC:lithium:SEI=2:2:1     for SEI=(CH2OCO2Li)2
        Ratio_CeEC_JR  = TotECMol_JR    /   Vol_Elely_JR_new   / c_EC_JR_old
        Ratio_CeLi_JR  = TotLi_Elely_JR_New    /   TotLi_Elely_JR_Old   /  Ratio_Dryout # Mark, change on 21-11-19
        c_e_r_new  = c_e_r_old;
        c_EC_r_new = c_EC_r_old;
        Width_new   = Ratio_Dryout * L_y;
        
    # Collect the above parameter in Data_Pack to shorten the code  
    Data_Pack   = [
        Vol_EC_consumed, 
        Vol_Elely_need, 
        Test_V, 
        Test_V2, 
        Vol_Elely_add, 
        Vol_Elely_Tot_new, 
        Vol_Elely_JR_new, 
        Vol_Pore_tot_new, 
        Vol_Pore_decrease, 
        c_e_r_new, c_EC_r_new,
        Ratio_Dryout, Ratio_CeEC_JR, 
        Ratio_CeLi_JR,
        Width_new, 
        ]  # 16 in total
    #print('Loss of lithium to negative SEI', LLINegSEI, 'mol') 
    #print('Loss of lithium to dead lithium plating', LLINegDeadLiP, 'mol') 
    #############################################################################################################################  
    ###################       Step-4 Update parameters here        ##############################################################
    #############################################################################################################################
    Para.update(   
        {'Bulk solvent concentration [mol.m-3]':  
         c_EC_JR_old * Ratio_CeEC_JR  })
    Para.update(
        {'EC initial concentration in electrolyte [mol.m-3]':
         c_EC_JR_old * Ratio_CeEC_JR },) 
    Para.update(   
        {'Ratio of Li-ion concentration change in electrolyte consider solvent consumption':  
        Ratio_CeLi_JR }, check_already_exists=False) 
    Para.update(   
        {'Current total electrolyte volume in whole cell [m3]':  Vol_Elely_Tot_new  }, 
        check_already_exists=False)
    Para.update(   
        {'Current total electrolyte volume in jelly roll [m3]':  Vol_Elely_JR_new  }, 
        check_already_exists=False)
    Para.update(   
        {'Ratio of electrolyte dry out in jelly roll':Ratio_Dryout}, 
        check_already_exists=False)
    Para.update(   {'Electrode width [m]':Width_new})    
    Para.update(   
        {'Current solvent concentration in the reservoir [mol.m-3]':c_EC_r_new}, 
        check_already_exists=False)     
    Para.update(   
        {'Current electrolyte concentration in the reservoir [mol.m-3]':c_e_r_new}, 
        check_already_exists=False)             
    return Data_Pack,Para

def Get_Last_state(Model, Sol):
    dict_short = {}
    list_short = []
    # update 220808 to satisfy random model option:
    for var, equation in Model.initial_conditions.items():
        #print(var._name)
        list_short.append(var._name)
    # delete Porosity times concentration and Electrolyte potential then add back
    list_short.remove("Porosity times concentration [mol.m-3]");
    list_short.remove("Electrolyte potential [V]");
    list_short.extend(
        ("Negative electrode porosity times concentration [mol.m-3]",
        "Separator porosity times concentration [mol.m-3]",
        "Positive electrode porosity times concentration [mol.m-3]",   

        "Negative electrolyte potential [V]",
        "Separator electrolyte potential [V]",
        "Positive electrolyte potential [V]",))
    for list_short_i in list_short:
        dict_short.update( { list_short_i : Sol.last_state[list_short_i].data  } )
    return list_short,dict_short


# Define a function to calculate based on previous solution
def Run_Model_Base_On_Last_Solution( 
    Model  , Sol , Para_update, ModelExperiment, 
    Update_Cycles,Temper_i ,mesh_list,submesh_strech, solver_type="Casadi"):
    # Use Sulzer's method: inplace = false
    # Important line: define new model based on previous solution
    Ratio_CeLi = Para_update[
        "Ratio of Li-ion concentration change in electrolyte consider solvent consumption"]
    if isinstance(Sol, pb.solvers.solution.Solution):
        list_short,dict_short = Get_Last_state(Model, Sol)
    elif isinstance(Sol, list):
        [dict_short, getSth] = Sol
        list_short = []
        for var, equation in Model.initial_conditions.items():
            list_short.append(var._name)
    else:
        print("!! Big problem, Sol here is neither solution or list")

    dict_short["Negative electrode porosity times concentration [mol.m-3]"] = (
        dict_short["Negative electrode porosity times concentration [mol.m-3]"] * Ratio_CeLi )# important: update sol here!
    dict_short["Separator porosity times concentration [mol.m-3]"] = (
        dict_short["Separator porosity times concentration [mol.m-3]"] * Ratio_CeLi )# important: update sol here!
    dict_short["Positive electrode porosity times concentration [mol.m-3]"] = (
        dict_short["Positive electrode porosity times concentration [mol.m-3]"] * Ratio_CeLi )# important: update sol here!
    Model_new = Model.set_initial_conditions_from(dict_short, inplace=False)
    Para_update.update(   {'Ambient temperature [K]':Temper_i });   # run model at 45 degree C
    
    var = pb.standard_spatial_vars  
    var_pts = {
        var.x_n: int(mesh_list[0]),  
        var.x_s: int(mesh_list[1]),  
        var.x_p: int(mesh_list[2]),  
        var.r_n: int(mesh_list[3]),  
        var.r_p: int(mesh_list[4]),  
        }
    submesh_types = Model.default_submesh_types
    if submesh_strech == "nan":
        pass
    else:
        particle_mesh = pb.MeshGenerator(
            pb.Exponential1DSubMesh, 
            submesh_params={"side": "right", "stretch": int(submesh_strech)})
        submesh_types["negative particle"] = particle_mesh
        submesh_types["positive particle"] = particle_mesh 
    Call_Age = RioCallback()  # define callback
    
    # update 231208 - try 3 times until give up 
    i_run_try = 0
    str_err = "Initialize only"
    while i_run_try<3:
        try:
            if i_run_try < 2:
                Simnew = pb.Simulation(
                    Model_new,
                    experiment = ModelExperiment, 
                    parameter_values=Para_update, 
                    solver = get_solver(solver_type),
                    var_pts = var_pts,
                    submesh_types=submesh_types )
                Sol_new = Simnew.solve(
                    calc_esoh=False,
                    save_at_cycles = Update_Cycles,
                    callbacks=Call_Age)
                if Call_Age.success == False:
                    raise Experiment_error_infeasible("Self detect")
            else:   # i_run_try = 2, 
                Simnew = pb.Simulation(
                    Model_new,
                    experiment = ModelExperiment, 
                    parameter_values=Para_update, 
                    solver = get_solver(solver_type, return_solution_if_failed_early=True),
                    var_pts = var_pts,
                    submesh_types=submesh_types )
                Sol_new = Simnew.solve(
                    calc_esoh=False,
                    # save_at_cycles = Update_Cycles,
                    callbacks=Call_Age)
                Succeed_AGE_cycs = len(Sol_new.cycles)
                if Succeed_AGE_cycs < Update_Cycles:
                    str_err = f"Partially succeed to run the ageing set for {Succeed_AGE_cycs} cycles the {i_run_try+1}th time"
                    print(str_err)
                else: 
                    str_err = f"Fully succeed to run the ageing set for {Succeed_AGE_cycs} cycles the {i_run_try+1}th time"
                    print(str_err)
        except (
            pb.expression_tree.exceptions.ModelError,
            pb.expression_tree.exceptions.SolverError
            ) as e:
            i_run_try += 1
            Sol_new = "Model error or solver error"
            str_err = f"Fail to run the ageing set due to {Sol_new} for the {i_run_try}th time"
            print(str_err)
            DeBug_List = [ Para_update, Update_Cycles, dict_short, str_err ]
        except Experiment_error_infeasible as custom_error:
            i_run_try += 1
            Sol_new = "Experiment error or infeasible"
            DeBug_List = [
                Model, Model_new, Call_Age, Simnew,  Sol , Sol_new, Para_update, ModelExperiment, 
                Update_Cycles,Temper_i ,mesh_list,submesh_strech, var_pts,
                submesh_types,  list_short, dict_short, 
            ]
            str_err= f"{Sol_new}: {custom_error} for ageing set for the {i_run_try}th time"
            print(str_err)
            DeBug_List = [ Para_update, Update_Cycles, dict_short, str_err ]
        else:
            i_run_try += 1
            # add 230221 - update 230317 try to access Throughput capacity more than once
            i_try = 0
            while i_try<3:
                try:
                    getSth2 = Sol_new['Throughput capacity [A.h]'].entries[-1]
                except:
                    i_try += 1
                    print(f"Fail to read Throughput capacity for the {i_try}th time")
                else:
                    break
            # update 240110
            if isinstance(Sol, pb.solvers.solution.Solution):
                i_try = 0
                while i_try<3:
                    try:
                        getSth = Sol['Throughput capacity [A.h]'].entries[-1]
                    except:
                        i_try += 1
                        print(f"Fail to read Throughput capacity for the {i_try}th time")
                    else:
                        break
            elif isinstance(Sol, list):
                print("Must be the first run after restart, already have getSth")
            else:
                print("!! Big problem, Sol here is neither solution or list")
            # update 23-11-16 change method to get throughput capacity:
            # # (1) add old ones, as before; (2): change values of the last one
            Sol_new['Throughput capacity [A.h]'].entries += getSth
            cyc_number = len(Sol_new.cycles)
            if not Update_Cycles == 1: # the solution is imcomplete in this case
                thr_1st = np.trapz(
                    abs(Sol_new.cycles[0]["Current [A]"].entries), 
                    Sol_new.cycles[0]["Time [h]"].entries) # in A.h
                thr_end = np.trapz(
                    abs(Sol_new.cycles[-1]["Current [A]"].entries), 
                    Sol_new.cycles[-1]["Time [h]"].entries) # in A.h
                thr_tot = (thr_1st+thr_end) / 2 * cyc_number
            else:
                thr_tot = abs(
                    Sol_new['Throughput capacity [A.h]'].entries[-1] - 
                    Sol_new['Throughput capacity [A.h]'].entries[0]  )
            Sol_new['Throughput capacity [A.h]'].entries[-1] = getSth + thr_tot # only the last one is true
            DeBug_List = [ Para_update, Update_Cycles, dict_short, str_err ]
            print(f"Succeed to run the ageing set for {cyc_number} cycles the {i_run_try}th time")
            break # terminate the loop of trying to solve if you can get here. 
    
    # print("Solved this model in {}".format(ModelTimer.time()))
    Result_list = [Model_new, Sol_new,Call_Age,DeBug_List]
    return Result_list

# Update 231117 new method to get throughput capacity to avoid problems of empty solution
def Get_ThrCap(sol_PRT): 
    thr_tot = 0
    for cycle in sol_PRT.cycles:
        for step in cycle.steps:
            # print(type(step))
            if not isinstance(step,pb.solvers.solution.EmptySolution):
                thr_i = np.trapz(
                    abs(step["Current [A]"].entries), 
                    step["Time [h]"].entries)
                thr_tot += thr_i
    return thr_tot

def Run_Model_Base_On_Last_Solution_RPT( 
    Model  , Sol,  Para_update, 
    ModelExperiment ,Update_Cycles, Temper_i,mesh_list,submesh_strech, solver_type="Casadi"):
    # Use Sulzer's method: inplace = false
    Ratio_CeLi = Para_update["Ratio of Li-ion concentration change in electrolyte consider solvent consumption"]
    # print("Model is now using average EC Concentration of:",Para_update['Bulk solvent concentration [mol.m-3]'])
    # print("Ratio of electrolyte dry out in jelly roll is:",Para_update['Ratio of electrolyte dry out in jelly roll'])
    # print("Model is now using an electrode width of:",Para_update['Electrode width [m]'])
    # Important line: define new model based on previous solution
    list_short,dict_short = Get_Last_state(Model, Sol)
    dict_short["Negative electrode porosity times concentration [mol.m-3]"] = (
        dict_short["Negative electrode porosity times concentration [mol.m-3]"] * Ratio_CeLi )# important: update sol here!
    dict_short["Separator porosity times concentration [mol.m-3]"] = (
        dict_short["Separator porosity times concentration [mol.m-3]"] * Ratio_CeLi )# important: update sol here!
    dict_short["Positive electrode porosity times concentration [mol.m-3]"] = (
        dict_short["Positive electrode porosity times concentration [mol.m-3]"] * Ratio_CeLi )# important: update sol here!
    Model_new = Model.set_initial_conditions_from(dict_short, inplace=False)
    Para_update.update(   {'Ambient temperature [K]':Temper_i });
    
    var = pb.standard_spatial_vars  
    var_pts = {
        var.x_n: int(mesh_list[0]),  
        var.x_s: int(mesh_list[1]),  
        var.x_p: int(mesh_list[2]),  
        var.r_n: int(mesh_list[3]),  
        var.r_p: int(mesh_list[4]),  }
    submesh_types = Model.default_submesh_types
    if submesh_strech == "nan":
        pass
    else:
        particle_mesh = pb.MeshGenerator(
            pb.Exponential1DSubMesh, 
            submesh_params={"side": "right", "stretch": int(submesh_strech)})
        submesh_types["negative particle"] = particle_mesh
        submesh_types["positive particle"] = particle_mesh 
    Simnew = pb.Simulation(
        Model_new,
        experiment = ModelExperiment, 
        parameter_values=Para_update, 
        solver = get_solver(solver_type),
        var_pts = var_pts,
        submesh_types=submesh_types
    )
    Call_RPT = RioCallback()  # define callback
    # update 231208 - try 3 times until give up 
    i_run_try = 0
    while i_run_try<3:
        try:
            Sol_new = Simnew.solve(
                calc_esoh=False,
                # save_at_cycles = Update_Cycles,
                callbacks=Call_RPT) 
            if Call_RPT.success == False:
                raise Experiment_error_infeasible("Self detect")        
        except (
            pb.expression_tree.exceptions.ModelError,
            pb.expression_tree.exceptions.SolverError
            ) as e:
            i_run_try += 1
            Sol_new = "Model error or solver error"
            str_err = f"Fail to run RPT due to {Sol_new} for the {i_run_try}th time"
            print(str_err)
            DeBug_List = [ Para_update, Update_Cycles, dict_short, str_err ]
        except Experiment_error_infeasible as custom_error:
            i_run_try += 1
            Sol_new = "Experiment error or infeasible"
            DeBug_List = [
                Model, Model_new, Call_RPT, Simnew,   Sol , Sol_new, Para_update, ModelExperiment, 
                Update_Cycles,Temper_i ,mesh_list,submesh_strech, var_pts,
                submesh_types,  list_short, dict_short, 
            ]
            str_err = f"{Sol_new}: {custom_error} for RPT for the {i_run_try}th time"
            print(str_err)
            DeBug_List = [ Para_update, Update_Cycles, dict_short, str_err ]
        else:
            # add 230221 - update 230317 try to access Throughput capacity more than once
            i_run_try += 1
            i_try = 0
            while i_try<3:
                try:
                    getSth2 = Sol_new['Throughput capacity [A.h]'].entries[-1]
                except:
                    i_try += 1
                    print(f"Fail to read Throughput capacity for the {i_try}th time")
                else:
                    break
            i_try = 0
            while i_try<3:
                try:
                    getSth = Sol['Throughput capacity [A.h]'].entries[-1]
                except:
                    i_try += 1
                    print(f"Fail to read Throughput capacity for the {i_try}th time")
                else:
                    break
            Sol_new['Throughput capacity [A.h]'].entries += getSth
            # update 23-11-17 change method to get throughput capacity to avioid problems of empty solution:
            thr_tot = Get_ThrCap(Sol_new)
            Sol_new['Throughput capacity [A.h]'].entries[-1] = getSth + thr_tot # only the last one is true
            DeBug_List = "Empty"
            print(f"Succeed to run RPT for the {i_run_try}th time")
            break # terminate the loop of trying to solve if you can get here. 
    # print("Solved this model in {}".format(ModelTimer.time()))
    Result_List_RPT = [Model_new, Sol_new,Call_RPT,DeBug_List]
    return Result_List_RPT





# Update 2023-10-04 
def Overwrite_Initial_L_SEI_0_Neg_Porosity(Para_0,cap_loss):
    """ 
    This is to overwrite the initial negative electrode porosity 
    and initial SEI thickness (inner, outer) to be consistent 
    with the initial capacity loss 
    """
    delta_Q_SEI = cap_loss * 3600
    V_SEI = Para_0["Outer SEI partial molar volume [m3.mol-1]"] 
    # do this when finish updating
    F = 96485.3
    A = Para_0["Electrode width [m]"] * Para_0["Electrode height [m]"]
    z_SEI = Para_0["Ratio of lithium moles to SEI moles"] 
    L_neg = Para_0["Negative electrode thickness [m]"] 
    eps_act_neg = Para_0["Negative electrode active material volume fraction"]
    R_neg =   Para_0["Negative particle radius [m]"]
    l_cr_init = Para_0["Negative electrode initial crack length [m]"]
    w_cr = Para_0["Negative electrode initial crack width [m]"]
    rho_cr = Para_0["Negative electrode number of cracks per unit area [m-2]"]
    a_neg = (3 * eps_act_neg / R_neg)
    roughness = 1 + 2 * l_cr_init * w_cr * rho_cr
    L_SEI_init = delta_Q_SEI  * V_SEI / (
        z_SEI * F * A * L_neg * a_neg * roughness)

    delta_epi = (L_SEI_init ) * roughness * a_neg
    L_inner_init = L_SEI_init / 2
    epi = 0.25 - delta_epi
    # print(L_inner_init,epi)
    # important: update here!
    Para_0["Negative electrode porosity"] = epi
    Para_0["Initial outer SEI thickness [m]"] = L_inner_init + 2.5e-9
    Para_0["Initial inner SEI thickness [m]"] = L_inner_init + 2.5e-9
    print(f"Has Overwritten Initial outer SEI thickness [m] to be {(L_inner_init+2.5e-9):.2e} and Negative electrode porosity to be {epi:.3f} to account for initial capacity loss of {cap_loss:.3f} Ah")

    return Para_0

# this function is to initialize the para with a known dict
def Para_init(Para_dict,cap_st):
    Para_dict_used = Para_dict.copy()

    Para_0=pb.ParameterValues(Para_dict_used["Para_Set"]  )
    Para_dict_used.pop("Para_Set")
    import ast,json

    if Para_dict_used.__contains__("Total ageing cycles"):
        Total_Cycles = Para_dict_used["Total ageing cycles"]  
        Para_dict_used.pop("Total ageing cycles")
    if Para_dict_used.__contains__("Ageing cycles between RPT"):
        Cycle_bt_RPT = Para_dict_used["Ageing cycles between RPT"]  
        Para_dict_used.pop("Ageing cycles between RPT")
    if Para_dict_used.__contains__("Update cycles for ageing"):
        Update_Cycles = Para_dict_used["Update cycles for ageing"]  
        Para_dict_used.pop("Update cycles for ageing")
    if Para_dict_used.__contains__("Cycles within RPT"):
        RPT_Cycles = Para_dict_used["Cycles within RPT"]  
        Para_dict_used.pop("Cycles within RPT")
    if Para_dict_used.__contains__("Ageing temperature"):
        Temper_i = Para_dict_used["Ageing temperature"]  + 273.15 # update: change to K
        Para_dict_used.pop("Ageing temperature")
    if Para_dict_used.__contains__("RPT temperature"):
        Temper_RPT = Para_dict_used["RPT temperature"] + 273.15 # update: change to K 
        Para_dict_used.pop("RPT temperature")
    if Para_dict_used.__contains__("Mesh list"):
        mesh_list = Para_dict_used["Mesh list"]
        mesh_list = json.loads(mesh_list)  
        Para_dict_used.pop("Mesh list")
    if Para_dict_used.__contains__("Exponential mesh stretch"):
        submesh_strech = Para_dict_used["Exponential mesh stretch"]  
        Para_dict_used.pop("Exponential mesh stretch")
    else:
        submesh_strech = "nan";
    if Para_dict_used.__contains__("Model option"):
        model_options = Para_dict_used["Model option"] 
        # careful: need to change str to dict 
        model_options = ast.literal_eval(model_options)
        Para_dict_used.pop("Model option")

    cap_increase = 0
    
    if Para_dict_used.__contains__("Initial Neg SOC"):
        c_Neg1SOC_in = (
            Para_0["Maximum concentration in negative electrode [mol.m-3]"]
            *Para_dict_used["Initial Neg SOC"]  )
        Para_0.update(
            {"Initial concentration in negative electrode [mol.m-3]":
            c_Neg1SOC_in})
        Para_dict_used.pop("Initial Neg SOC")
    if Para_dict_used.__contains__("Initial Pos SOC"):    
        c_Pos1SOC_in = (
            Para_0["Maximum concentration in positive electrode [mol.m-3]"]
            *Para_dict_used["Initial Pos SOC"]  )
        Para_0.update(
            {"Initial concentration in positive electrode [mol.m-3]":
            c_Pos1SOC_in})
        Para_dict_used.pop("Initial Pos SOC")
    # 'Outer SEI partial molar volume [m3.mol-1]'
    if Para_dict_used.__contains__(
        "Outer SEI partial molar volume [m3.mol-1]"): 
        Para_0.update(
            {"Outer SEI partial molar volume [m3.mol-1]":
            Para_dict_used["Outer SEI partial molar volume [m3.mol-1]"]})
        Para_0.update(
            {"Inner SEI partial molar volume [m3.mol-1]":
            Para_dict_used["Outer SEI partial molar volume [m3.mol-1]"]})
        
        Para_dict_used.pop("Outer SEI partial molar volume [m3.mol-1]")


    CyclePack = [ 
        Total_Cycles,Cycle_bt_RPT,Update_Cycles,RPT_Cycles,
        Temper_i,Temper_RPT,mesh_list,submesh_strech,
        model_options,     cap_increase]
    # Mark Ruihe - updated 230222 - from P3
    if "Solver_Type" in Para_dict_used:
        Para_dict_used.pop("Solver_Type")
        
    for key, value in Para_dict_used.items():
        # risk: will update parameter that doesn't exist, 
        # so need to make sure the name is right 
        if isinstance(value, str):
            Para_0.update({key: eval(value)})
            #Para_dict_used.pop(key)
        else:
            Para_0.update({key: value},check_already_exists=False)




    cap_loss =   5.0 - 4.86491 # cap_st      # change for debug=   4.86491  
    Para_0 = Overwrite_Initial_L_SEI_0_Neg_Porosity(Para_0,cap_loss)



    return CyclePack,Para_0

# Add 220808 - to simplify the post-processing
def GetSol_dict (my_dict, keys_all, Sol, 
    cycle_no,step_CD, step_CC , step_RE, step_CV ):

    [keys_loc,keys_tim,keys_cyc]  = keys_all;
    # get time_based variables:
    if len(keys_tim): 
        for key in keys_tim:
            step_no = eval("step_{}".format(key[0:2]))
            if key[3:] == "Time [h]":
                my_dict[key].append  (  (
                    Sol.cycles[cycle_no].steps[step_no][key[3:]].entries
                    -
                    Sol.cycles[cycle_no].steps[step_no][key[3:]].entries[0]).tolist()  )
            elif key[3:] == "Anode potential [V]":
                sol_step = Sol.cycles[cycle_no].steps[step_no]
                mesh_sep = len(sol_step["Separator electrolyte potential [V]"].entries[:,0])
                if mesh_sep % 2 ==0:
                    phi_ref = (
                        sol_step["Separator electrolyte potential [V]"].entries[mesh_sep//2-1,:]
                        +
                        sol_step["Separator electrolyte potential [V]"].entries[mesh_sep//2+1,:]
                        ) / 2
                    #print(mesh_sep/2+0.5)
                else:
                    phi_ref = sol_step["Separator electrolyte potential [V]"].entries[mesh_sep//2,:]
                V_n = -phi_ref 
                my_dict[key].append(  V_n.tolist()  )
            elif key[3:] == "Cathode potential [V]":
                sol_step = Sol.cycles[cycle_no].steps[step_no]
                mesh_sep = len(sol_step["Separator electrolyte potential [V]"].entries[:,0])
                if mesh_sep % 2 ==0:
                    phi_ref = (
                        sol_step["Separator electrolyte potential [V]"].entries[mesh_sep//2-1,:]
                        +
                        sol_step["Separator electrolyte potential [V]"].entries[mesh_sep//2+1,:]
                        ) / 2
                    #print(mesh_sep/2+0.5)
                else:
                    phi_ref = sol_step["Separator electrolyte potential [V]"].entries[mesh_sep//2,:]
                V =  sol_step["Terminal voltage [V]"].entries
                V_p = V - phi_ref   
                my_dict[key].append(  V_p.tolist()  )
            else:
                my_dict[key].append(  (
                    Sol.cycles[cycle_no].steps[step_no][key[3:]].entries).tolist()  )
    # get cycle_step_based variables: # isn't an array
    if len(keys_cyc): 
        for key in keys_cyc:
            if key in ["Discharge capacity [A.h]"]:
                step_no = step_CD;
                my_dict[key].append  (
                    Sol.cycles[cycle_no].steps[step_no][key].entries[-1]
                    - 
                    Sol.cycles[cycle_no].steps[step_no][key].entries[0])
            elif key in ["Throughput capacity [A.h]"]: # 
                i_try = 0
                while i_try<3:
                    try:
                        getSth = Sol[key].entries[-1]
                    except:
                        i_try += 1
                        print(f"Fail to read Throughput capacity for the {i_try}th time")
                    else:
                        break
                my_dict[key].append(abs(getSth))
            elif key[0:5] in ["CDend","CCend","CVend","REend",
                "CDsta","CCsta","CVsta","REsta",]:
                step_no = eval("step_{}".format(key[0:2]))
                if key[2:5] == "sta":
                    my_dict[key].append  (
                        Sol.cycles[cycle_no].steps[step_no][key[6:]].entries[0])
                elif key[2:5] == "end":
                    my_dict[key].append  (
                        Sol.cycles[cycle_no].steps[step_no][key[6:]].entries[-1])
                
    # get location_based variables:
    if len(keys_loc): 
        for key in keys_loc:
            if key in ["x_n [m]","x [m]","x_s [m]","x_p [m]"]:
                #print("These variables only add once")
                if not len(my_dict[key]):   # special: add only once
                    my_dict[key] = (Sol[key].entries[:,-1]).tolist()
            elif key[0:5] in ["CDend","CCend","CVend","REend",
                            "CDsta","CCsta","CVsta","REsta",]:      
                #print("These variables add multiple times")
                step_no = eval("step_{}".format(key[0:2]))
                if key[2:5] == "sta":
                    my_dict[key].append  ((
                        Sol.cycles[cycle_no].steps[step_no][key[6:]].entries[:,0]).tolist() )
                elif key[2:5] == "end":
                    my_dict[key].append ( (
                        Sol.cycles[cycle_no].steps[step_no][key[6:]].entries[:,-1]).tolist()  )
    return my_dict                              

def Get_SOH_LLI_LAM(my_dict_RPT,model_options,DryOut,mdic_dry,cap_0):
    my_dict_RPT['Throughput capacity [kA.h]'] = (
        np.array(my_dict_RPT['Throughput capacity [A.h]'])/1e3).tolist()
    my_dict_RPT['CDend SOH [%]'] = ((
        np.array(my_dict_RPT["Discharge capacity [A.h]"])
        /
        cap_0       # my_dict_RPT["Discharge capacity [A.h]"][0] # Mark: change to this so that every case has same standard for reservior paper
        )*100).tolist()
    my_dict_RPT["CDend LAM_ne [%]"] = ((1-
        np.array(my_dict_RPT['CDend Negative electrode capacity [A.h]'])
        /my_dict_RPT['CDend Negative electrode capacity [A.h]'][0])*100).tolist()
    my_dict_RPT["CDend LAM_pe [%]"] = ((1-
        np.array(my_dict_RPT['CDend Positive electrode capacity [A.h]'])
        /my_dict_RPT['CDend Positive electrode capacity [A.h]'][0])*100).tolist()
    my_dict_RPT["CDend LLI [%]"] = ((1-
        np.array(my_dict_RPT["CDend Total lithium capacity in particles [A.h]"])
        /my_dict_RPT["CDend Total lithium capacity in particles [A.h]"][0])*100).tolist()
    if model_options.__contains__("SEI"):
        my_dict_RPT["CDend LLI SEI [%]"] = ((
            np.array(
                my_dict_RPT["CDend Loss of capacity to SEI [A.h]"]-
                my_dict_RPT["CDend Loss of capacity to SEI [A.h]"][0]
                )
            /my_dict_RPT["CDend Total lithium capacity in particles [A.h]"][0])*100).tolist()
    else:
        my_dict_RPT["CDend LLI SEI [%]"] = (
            np.zeros(
                np.size(
                    my_dict_RPT["CDend LLI [%]"]))).tolist()
    if model_options.__contains__("SEI on cracks"):
        my_dict_RPT["CDend LLI SEI on cracks [%]"] = ((
            np.array(
                my_dict_RPT["CDend Loss of capacity to SEI on cracks [A.h]"]-
                my_dict_RPT["CDend Loss of capacity to SEI on cracks [A.h]"][0]
                )
            /my_dict_RPT["CDend Total lithium capacity in particles [A.h]"][0])*100).tolist()
    else:
        my_dict_RPT["CDend LLI SEI on cracks [%]"] = (
            np.zeros(
                np.size(
                    my_dict_RPT["CDend LLI [%]"]))).tolist()
    if model_options.__contains__("lithium plating"):
        my_dict_RPT["CDend LLI lithium plating [%]"] = ((
            np.array(
                my_dict_RPT["CDend Loss of capacity to lithium plating [A.h]"]-
                my_dict_RPT["CDend Loss of capacity to lithium plating [A.h]"][0]
                )
            /my_dict_RPT["CDend Total lithium capacity in particles [A.h]"][0])*100).tolist()
    else:
        my_dict_RPT["CDend LLI lithium plating [%]"] =  (
            np.zeros(
                np.size(
                    my_dict_RPT["CDend LLI [%]"]))).tolist()    
    my_dict_RPT["CDend LLI due to LAM [%]"] = (
        np.array(my_dict_RPT['CDend LLI [%]'])
        -my_dict_RPT["CDend LLI SEI [%]"]
        -my_dict_RPT["CDend LLI lithium plating [%]"]
        -my_dict_RPT["CDend LLI SEI on cracks [%]"] )
    if DryOut == "On":
        LAM_to_Dry   = 100-np.array(
            mdic_dry['Width_all']) /mdic_dry['Width_all'][0]*100
        LAM_to_Dry_end=LAM_to_Dry[-1]
    else:
        LAM_to_Dry_end = 0
    LAM_to_Crack_NE_end=my_dict_RPT['CDend LAM_ne [%]'][-1]-LAM_to_Dry_end
    LAM_to_Crack_PE_end=my_dict_RPT['CDend LAM_pe [%]'][-1]-LAM_to_Dry_end
    my_dict_RPT["LAM_to_Dry [%] end"] = LAM_to_Dry_end
    my_dict_RPT["LAM_to_Crack_NE [%] end"] = LAM_to_Crack_NE_end
    my_dict_RPT["LAM_to_Crack_PE [%] end"] = LAM_to_Crack_PE_end
    return my_dict_RPT


def get_solver(solver_type="Casadi", return_solution_if_failed_early=False):
    if solver_type == "IDAKLU":
        # IDAKLU specific options
        options = {
            "max_num_steps": 100000,
            "num_steps_per_output": 100,
            "jacobian": "sparse",
        }
        return pb.IDAKLUSolver(atol=1e-8, rtol=1e-10, options=options)
    else:
        if return_solution_if_failed_early:
             return pb.CasadiSolver(mode="safe", return_solution_if_failed_early=True)
        else:
             return pb.CasadiSolver(mode="safe")

# define the model and run break-in cycle - 
# input parameter: model_options, Experiment_Breakin, Para_0, mesh_list, submesh_strech
# output: Sol_0 , Model_0, Call_Breakin
def Run_Breakin(
    model_options, Experiment_Breakin, 
    Para_0, mesh_list, submesh_strech,cap_increase, solver_type="Casadi"):

    Model_0 = pb.lithium_ion.DFN(options=model_options)
    # update 220926 - add diffusivity and conductivity as variables:
    c_e = Model_0.variables["Electrolyte concentration [mol.m-3]"]
    T = Model_0.variables["Cell temperature [K]"]
    D_e = Para_0["Electrolyte diffusivity [m2.s-1]"]
    sigma_e = Para_0["Electrolyte conductivity [S.m-1]"]
    Model_0.variables["Electrolyte diffusivity [m2.s-1]"] = D_e(c_e, T)
    Model_0.variables["Electrolyte conductivity [S.m-1]"] = sigma_e(c_e, T)
    var = pb.standard_spatial_vars  
    var_pts = {
        var.x_n: int(mesh_list[0]),  
        var.x_s: int(mesh_list[1]),  
        var.x_p: int(mesh_list[2]),  
        var.r_n: int(mesh_list[3]),  
        var.r_p: int(mesh_list[4]),  }       
    submesh_types = Model_0.default_submesh_types
    if submesh_strech == "nan":
        pass
    else:
        particle_mesh = pb.MeshGenerator(
            pb.Exponential1DSubMesh, 
            submesh_params={"side": "right", "stretch": submesh_strech})
        submesh_types["negative particle"] = particle_mesh
        submesh_types["positive particle"] = particle_mesh 


    """ exp_breakin_text = [ (
        # refill
        "Rest for 10 s",  
        #f"Hold at 4.2V until C/100",
        #"Rest for 1 hours (20 minute period)", 
        ) ] 
    Experiment_Breakin= pb.Experiment( 
            exp_breakin_text * 1) 
    Sim_0    = pb.Simulation(
        Model_0,        experiment = Experiment_Breakin,
        parameter_values = Para_0,
        solver = pb.CasadiSolver(),
        var_pts=var_pts,
        submesh_types=submesh_types) 
    Call_Breakin = RioCallback()
    Sol_0    = Sim_0.solve(calc_esoh=False,callbacks=Call_Breakin) """
    # generate a list of capacity increase perturbation:
    import random
    Cap_in_perturbation = []
    try_no = 4
    for k in range(try_no):
        random_number = 0
        while abs(random_number) <= 1e-6:
            random_number = random.uniform(-1e-5, 1e-5)
        Cap_in_perturbation.append(random_number)
    Cap_in_perturbation[0] = 0.0 # first time must be zero
    Cap_in_perturbation[1] = 3.9116344182112036E-4
    c_s_neg_baseline = Para_0["Initial concentration in negative electrode [mol.m-3]"]
    # update 240603 - try shift neg soc 4 times until give up 
    i_run_try = 0
    while i_run_try<try_no:
        try:
            Sim_0    = pb.Simulation(
                Model_0,        experiment = Experiment_Breakin,
                parameter_values = Para_0,
                solver = get_solver(solver_type),
                var_pts=var_pts,
                submesh_types=submesh_types) 
            Call_Breakin = RioCallback()    
            Sol_0    = Sim_0.solve(calc_esoh=False,callbacks=Call_Breakin)
        except (
            pb.expression_tree.exceptions.ModelError,
            pb.expression_tree.exceptions.SolverError
            ) as e:
            Sol_0 = "Model error or solver error"
            str_err = (
                f"Fail to run break in due to {Sol_0} for the {i_run_try}th time"
                f" with capacity increase of {cap_increase}Ah and "
                f"perturbation of {Cap_in_perturbation[i_run_try]:.2e}Ah")
            print();print(str_err);print()
            i_run_try += 1
        else:
            str_err = (
                f"Succeed to run break in for the {i_run_try}th time "
                f"with capacity increase of {cap_increase}Ah and "
                f"perturbation of {Cap_in_perturbation[i_run_try]:.2e}Ah")
            print();print(str_err);print()
            break
    Result_list_breakin = [Model_0,Sol_0,Call_Breakin]

    return Result_list_breakin

# Input: Para_0
# Output: mdic_dry, Para_0
def Initialize_mdic_dry(Para_0,Int_ElelyExces_Ratio):
    mdic_dry = {
        "CeEC_All": [],
        "c_EC_r_new_All": [],
        "c_e_r_new_All": [],
        "Ratio_CeEC_All":[],
        "Ratio_CeLi_All":[],
        "Ratio_Dryout_All":[],

        "Vol_Elely_Tot_All": [],
        "Vol_Elely_JR_All":[],
        "Vol_Pore_tot_All": [],        
        "Vol_EC_consumed_All":[],
        "Vol_Elely_need_All":[],
        "Width_all":[],
        "Vol_Elely_add_All":[],
        "Vol_Pore_decrease_All":[],
        "Test_V_All":[],
        "Test_V2_All":[],
    }
    T_0                  =  Para_0['Initial temperature [K]']
    Porosity_Neg_0       =  Para_0['Negative electrode porosity']  
    Porosity_Pos_0       =  Para_0['Positive electrode porosity']  
    Porosity_Sep_0       =  Para_0['Separator porosity']  
    cs_Neg_Max           =  Para_0["Maximum concentration in negative electrode [mol.m-3]"];
    L_p                  =  Para_0["Positive electrode thickness [m]"]
    L_n                  =  Para_0["Negative electrode thickness [m]"]
    L_s                  =  Para_0["Separator thickness [m]"]
    L_y                  =  Para_0["Electrode width [m]"]
    Para_0.update({'Initial Electrode width [m]':L_y}, check_already_exists=False)
    L_y_0                =  Para_0["Initial Electrode width [m]"]
    L_z                  =  Para_0["Electrode height [m]"]
    Para_0.update({'Initial Electrode height [m]':L_z}, check_already_exists=False)
    L_z_0                =  Para_0["Initial Electrode height [m]"]
    
    Vol_Elely_Tot        = (
        ( L_n*Porosity_Neg_0 +  L_p*Porosity_Pos_0  +  L_s*Porosity_Sep_0  )  
        * L_y_0 * L_z_0 * Int_ElelyExces_Ratio
     ) # Set initial electrolyte amount [L] 
    Vol_Elely_JR         =  (
        ( L_n*Porosity_Neg_0 +  L_p*Porosity_Pos_0  +  L_s*Porosity_Sep_0  )  
        * L_y_0 * L_z_0 )
    Vol_Pore_tot         =  (
        ( L_n*Porosity_Neg_0 +  L_p*Porosity_Pos_0  +  L_s*Porosity_Sep_0  )  
        * L_y_0 * L_z_0 )
    Ratio_CeEC           =  1.0; Ratio_CeLi =  1.0  ;Ratio_Dryout         =  1.0
    Vol_EC_consumed      =  0;Vol_Elely_need       =  0;Vol_Elely_add        =  0
    Vol_Pore_decrease    =  0;Test_V2 = 0; Test_V = 0;
    print('Initial electrolyte amount is ', Vol_Elely_Tot*1e6, 'mL') 
    Para_0.update(
        {'Current total electrolyte volume in jelly roll [m3]':
        Vol_Elely_JR}, check_already_exists=False)
    Para_0.update(
        {'Current total electrolyte volume in whole cell [m3]':
        Vol_Elely_Tot}, check_already_exists=False)   
    
    mdic_dry["Vol_Elely_Tot_All"].append(Vol_Elely_Tot*1e6);            
    mdic_dry["Vol_Elely_JR_All"].append(Vol_Elely_JR*1e6);     
    mdic_dry["Vol_Pore_tot_All"].append(Vol_Pore_tot*1e6);           
    mdic_dry["Ratio_CeEC_All"].append(Ratio_CeEC);                      
    mdic_dry["Ratio_CeLi_All"].append(Ratio_CeLi);             
    mdic_dry["Ratio_Dryout_All"].append(Ratio_Dryout);
    mdic_dry["Vol_EC_consumed_All"].append(Vol_EC_consumed*1e6);        
    mdic_dry["Vol_Elely_need_All"].append(Vol_Elely_need*1e6);     
    mdic_dry["Width_all"].append(L_y_0);
    mdic_dry["Vol_Elely_add_All"].append(Vol_Elely_add*1e6);            
    mdic_dry["Vol_Pore_decrease_All"].append(Vol_Pore_decrease*1e6);
    mdic_dry["Test_V_All"].append(Test_V*1e6); 
    mdic_dry["Test_V2_All"].append(Test_V2*1e6); 
    mdic_dry["c_e_r_new_All"].append(Para_0["Initial concentration in electrolyte [mol.m-3]"]); 
    mdic_dry["c_EC_r_new_All"].append(Para_0['EC initial concentration in electrolyte [mol.m-3]'])

    return mdic_dry,Para_0
    
def Update_mdic_dry(Data_Pack,mdic_dry):
    [
        Vol_EC_consumed, 
        Vol_Elely_need, 
        Test_V, 
        Test_V2, 
        Vol_Elely_add, 
        Vol_Elely_Tot_new, 
        Vol_Elely_JR_new, 
        Vol_Pore_tot_new, 
        Vol_Pore_decrease, 
        c_e_r_new, c_EC_r_new,
        Ratio_Dryout, Ratio_CeEC_JR, 
        Ratio_CeLi_JR,
        Width_new, ]= Data_Pack;
    mdic_dry["Vol_Elely_Tot_All"].append(Vol_Elely_Tot_new*1e6);            
    mdic_dry["Vol_Elely_JR_All"].append(Vol_Elely_JR_new*1e6);     
    mdic_dry["Vol_Pore_tot_All"].append(Vol_Pore_tot_new*1e6);           
    mdic_dry["Ratio_CeEC_All"].append(Ratio_CeEC_JR);                      
    mdic_dry["Ratio_CeLi_All"].append(Ratio_CeLi_JR);             
    mdic_dry["Ratio_Dryout_All"].append(Ratio_Dryout);
    mdic_dry["Vol_EC_consumed_All"].append(Vol_EC_consumed*1e6);        
    mdic_dry["Vol_Elely_need_All"].append(Vol_Elely_need*1e6);     
    mdic_dry["Width_all"].append(Width_new);
    mdic_dry["Vol_Elely_add_All"].append(Vol_Elely_add*1e6);            
    mdic_dry["Vol_Pore_decrease_All"].append(Vol_Pore_decrease*1e6);
    mdic_dry["Test_V_All"].append(Test_V*1e6); 
    mdic_dry["Test_V2_All"].append(Test_V2*1e6); 
    mdic_dry["c_e_r_new_All"].append(c_e_r_new); 
    mdic_dry["c_EC_r_new_All"].append(c_EC_r_new)

    return mdic_dry

# update 230312: add a function to get the discharge capacity and resistance
def Get_0p1s_R0(sol_RPT,Index,cap_full):
    Res_0p1s = []; SOC = [100,];
    for i,index in enumerate(Index):
        cycle = sol_RPT.cycles[index]
        Res_0p1s.append(   (
            np.mean(cycle.steps[1]["Terminal voltage [V]"].entries[-10:-1])
            - cycle.steps[2]["Terminal voltage [V]"].entries[0]
        ) / cycle.steps[2]["Current [A]"].entries[0] * 1000)
        if i > 0:
            Dis_Cap = abs(
                cycle.steps[2]["Discharge capacity [A.h]"].entries[0] 
                - cycle.steps[2]["Discharge capacity [A.h]"].entries[-1] )
            SOC.append(SOC[-1]-Dis_Cap/cap_full*100)
    return Res_0p1s[12],Res_0p1s,SOC

# Update 230517 add a function to get R_50%SOC from C/2 discharge
def Get_R_from_0P5C_CD(step_0P5C_CD,cap_full):
    # print("Total data points: ",len(step_0P5C_CD["Time [h]"].entries))
    Dis_Cap = abs(
        step_0P5C_CD["Discharge capacity [A.h]"].entries[0] 
        - step_0P5C_CD["Discharge capacity [A.h]"].entries )
    SOC_0p5C = (1-Dis_Cap/cap_full)*100
    #print(SOC_0p5C)
    V_ohmic = (
    step_0P5C_CD['Battery open-circuit voltage [V]'].entries 
    - step_0P5C_CD["Terminal voltage [V]"].entries
    + step_0P5C_CD["Battery particle concentration overpotential [V]"].entries 
    + step_0P5C_CD["X-averaged battery concentration overpotential [V]" ].entries
    )
    # print("Applied current [A]:",step_0P5C_CD["Current [A]"].entries[0])
    Res_0p5C = V_ohmic/step_0P5C_CD["Current [A]"].entries[0] * 1e3
    Res_0p5C_50SOC = np.interp(50,np.flip(SOC_0p5C),np.flip(Res_0p5C),)
    SOC_0p5C = SOC_0p5C.tolist()
    Res_0p5C = Res_0p5C.tolist()
    Res_0p5C_50SOC = Res_0p5C_50SOC.tolist()
    return Res_0p5C_50SOC,Res_0p5C,SOC_0p5C  #,Rohmic_CD_2



def Initialize_exp_text(V_max, V_min, Add_Rest):
    if Add_Rest:
        exp_AGE_text = [(
            f"Discharge at 1C until {V_min}V", 
            "Rest for 1 second", 
            f"Charge at 0.3C until {V_max}V",
            f"Hold at {V_max} V until C/100",
            ),  ]  # *  78
        step_AGE_CD =0;   step_AGE_CC =2;   step_AGE_CV =3;
    else:
        exp_AGE_text = [(
            f"Discharge at 1C until {V_min}V", 
            f"Charge at 0.3C until {V_max}V",
            f"Hold at {V_max} V until C/100",
            ),  ]  # *  78
        step_AGE_CD =0;   step_AGE_CC =1;   step_AGE_CV =2;

    # now for RPT: 
    exp_RPT_Need_TopUp = [ (
        f"Charge at 0.3C until {V_max}V",
        f"Hold at {V_max}V until C/100",
        "Rest for 1 hours",
        "Rest for 10 s",   # add here to correct values of step_0p1C_CD 
        # 0.1C cycle 
        f"Discharge at 0.1C until {V_min} V",  
        "Rest for 3 hours",  
        f"Charge at 0.1C until {V_max} V",
        f"Hold at {V_max}V until C/100",
        "Rest for 1 hours",
        # 0.5C cycle 
        f"Discharge at 0.5C until {V_min} V",  
        "Rest for 3 hours",
        f"Charge at 0.5C until {V_max} V",
        f"Hold at {V_max}V until C/100",
        # Update 23-11-17: add one more 0.5C cycle to increase throughput capacity
        f"Discharge at 0.5C until {V_min} V",  
        "Rest for 3 hours",
        f"Charge at 0.5C until {V_max} V",
        f"Hold at {V_max}V until C/100",   
        "Rest for 3 hours",  
        ) ] 
    exp_RPT_No_TopUp = [ (
        "Rest for 10 s",   # add here to correct values of step_0p1C_CD
        "Rest for 10 s",   # add here to correct values of step_0p1C_CD
        "Rest for 1 hours", 
        "Rest for 10 s",   # add here to correct values of step_0p1C_CD
        # 0.1C cycle 
        f"Discharge at 0.1C until {V_min} V",  
        "Rest for 3 hours",  
        f"Charge at 0.1C until {V_max} V",
        f"Hold at {V_max}V until C/100",
        "Rest for 1 hours",
        # 0.5C cycle 
        f"Discharge at 0.5C until {V_min} V",  
        "Rest for 3 hours",
        f"Charge at 0.5C until {V_max} V",
        f"Hold at {V_max}V until C/100",
        # Update 23-11-17: add one more 0.5C cycle to increase throughput capacity
        f"Discharge at 0.5C until {V_min} V",  
        "Rest for 3 hours",
        f"Charge at 0.5C until {V_max} V",
        f"Hold at {V_max}V until C/100",   
        "Rest for 3 hours",  
        ) ] 
    exp_breakin_text = [ (
        # refill
        #"Rest for 10 s",   # add here to correct values of step_0p1C_CD
        #f"Hold at {V_max}V until C/100",
        #"Rest for 10 s", # Mark Ruihe change ad hoc setting for LFP 
        f"Discharge at 0.5C until {V_max-0.2}V", # start from discharge as it is easier for unbalanced cells
        f"Charge at 0.3C until {V_max}V",
        f"Hold at {V_max}V until C/100",
        "Rest for 1 hours", 
        # 0.1C cycle 
        f"Discharge at 0.1C until {V_min} V",  
        "Rest for 3 hours",  
        f"Charge at 0.1C until {V_max} V",
        f"Hold at {V_max}V until C/100",
        "Rest for 1 hours",
        # 0.5C cycle 
        f"Discharge at 0.5C until {V_min} V",  
        "Rest for 3 hours",
        f"Charge at 0.5C until {V_max} V",
        f"Hold at {V_max}V until C/100",
        # Update 23-11-17: add one more 0.5C cycle to increase throughput capacity
        f"Discharge at 0.5C until {V_min} V",  
        "Rest for 3 hours",
        f"Charge at 0.5C until {V_max} V",
        f"Hold at {V_max}V until C/100",   
        "Rest for 3 hours",  
        ) ] 
    exp_RPT_GITT_text = [ (
        "Rest for 5 minutes (1 minute period)",  
        "Rest for 1.2 seconds (0.1 second period)",  
        f"Discharge at C/2 for 4.8 minutes or until {V_min}V (0.1 second period)",
        "Rest for 1 hour", # (5 minute period)  
        ) ]
    exp_refill = [ (
        f"Charge at 0.3C until {V_max}V",
        f"Hold at {V_max}V until C/100",
        "Rest for 1 hours", 
        ) ] 
    exp_adjust_before_age = [ (
        # just a place holder for now TODO 
        "Rest for 1 hours", 
        ) ] 
    exp_RPT_text = exp_RPT_No_TopUp 
    # step index for RPT
    step_0p1C_CD = 4; step_0p1C_CC = 6;   step_0p1C_RE =5;    
    step_0p5C_CD = 9;  
    Pack_return = [
        exp_AGE_text, step_AGE_CD, step_AGE_CC, step_AGE_CV,
        exp_breakin_text, exp_RPT_text, exp_RPT_GITT_text, 
        exp_refill,exp_adjust_before_age,
        step_0p1C_CD, step_0p1C_CC, step_0p1C_RE, step_0p5C_CD
        ] 
    return Pack_return

# update 231205: write a function to get tot_cyc,cyc_age,update
def Get_tot_cyc(Temp_K):
    print(f"Aging T is {Temp_K}degC")
    tot_cyc = 1170; cyc_age = 78; update = 78;    # actually 77 on cycler
    return tot_cyc,cyc_age,update


# Get keys for output
def Get_Output_Keys(Para_dict):
    model_options = Para_dict["Model option"] 
    keys_loc_RPT = [ # MAY WANT TO SELECT AGEING CYCLE later
        # Default output:
        "x [m]",
        "x_n [m]",
        "x_s [m]",
        "x_p [m]",
        # default: end; 
        "CCend Porosity",
        "CCend Negative electrode interfacial current density [A.m-2]",
        "CCend Electrolyte potential [V]",
        "CCend Electrolyte concentration [mol.m-3]",
        "CCend Negative electrode reaction overpotential [V]",
        "CCend Negative particle surface concentration [mol.m-3]",
        #"CCend Negative electrode roughness ratio",
        #"CCend Total SEI on cracks thickness [m]",

        "CDend Porosity",
        "CDend Negative electrode interfacial current density [A.m-2]",
        "CDend Electrolyte potential [V]",
        "CDend Electrolyte concentration [mol.m-3]",
        "CDend Negative electrode reaction overpotential [V]",
        "CDend Negative particle surface concentration [mol.m-3]",
        #"CDend Negative electrode roughness ratio",
        #"CDend Total SEI on cracks thickness [m]",
        #"REend Total SEI on cracks thickness [m]",
    ]
    keys_tim_RPT = [
        # default: CD
        "CD Time [h]",
        "CD Terminal voltage [V]",
        "CD Anode potential [V]",    # self defined
        "CD Cathode potential [V]",  # self defined
        "CC Time [h]",
        "CC Terminal voltage [V]",
        "CC Anode potential [V]",    # self defined
        "CC Cathode potential [V]",  # self defined
    ]
    keys_cyc_RPT = [   # default: CDend
        "Discharge capacity [A.h]",
        "Throughput capacity [A.h]",
        "CDend Total lithium capacity in particles [A.h]",
        "CDend Loss of capacity to lithium plating [A.h]",
        "CDend Loss of capacity to SEI [A.h]",
        "CDend Loss of capacity to SEI on cracks [A.h]",
        #"CDend X-averaged total SEI on cracks thickness [m]",
        #"CDend X-averaged negative electrode roughness ratio",
        "CDend Local ECM resistance [Ohm]",
        "CDsta Negative electrode stoichiometry", 
        "CDend Negative electrode stoichiometry",
        "CDsta Positive electrode stoichiometry", 
        "CDend Positive electrode stoichiometry",
        "CDend Negative electrode capacity [A.h]",
        "CDend Positive electrode capacity [A.h]",
    ]

    keys_loc_AGE = [ # MAY WANT TO SELECT AGEING CYCLE later
        # Default output:
        "x [m]",
        "x_n [m]",
        "x_s [m]",
        "x_p [m]",
        # default: end; 
        "CCend Porosity",
        "CCend Negative electrode interfacial current density [A.m-2]",
        "CCend Electrolyte potential [V]",
        "CCend Electrolyte concentration [mol.m-3]",
        "CCend Negative electrode reaction overpotential [V]",
        "CCend Negative particle surface concentration [mol.m-3]",
        "CCend Negative electrode surface potential difference [V]",
        "CCend SEI film overpotential [V]",
        #"CCend Negative electrode roughness ratio",
        #"CCend Total SEI on cracks thickness [m]",

        "CDend Porosity",
        "CDend Negative electrode interfacial current density [A.m-2]",
        "CDend Electrolyte potential [V]",
        "CDend Electrolyte concentration [mol.m-3]",
        "CDend Negative electrode reaction overpotential [V]",
        "CDend Negative particle surface concentration [mol.m-3]",
        #"CDend Negative electrode roughness ratio",
        #"CDend Total SEI on cracks thickness [m]",
        "CDend Negative electrode surface potential difference [V]",
        "CDend SEI film overpotential [V]",
        "CDend Electrolyte diffusivity [m2.s-1]",
        "CDend Electrolyte conductivity [S.m-1]",
    ]
    keys_tim_AGE = [
        # default: CD
        "CD Time [h]",
        "CD Terminal voltage [V]",
        "CD Anode potential [V]",    # self defined
        "CD Cathode potential [V]",  # self defined
        
        "CC Time [h]",
        "CC Terminal voltage [V]",
        "CC Anode potential [V]",    # self defined
        "CC Cathode potential [V]",  # self defined
    ]
    keys_cyc_AGE = [];
    keys_all_RPT = [keys_loc_RPT,keys_tim_RPT,keys_cyc_RPT]
    keys_all_AGE = [keys_loc_AGE,keys_tim_AGE,keys_cyc_AGE]
    keys_all = [keys_all_RPT,keys_all_AGE]
    return keys_all

def Run_Aging_Sim(
    Para_dict,  Path_List,  Re_No,        
    Timelimit,    Options): 

    ##########################################################
    ##############    Part-1: Initialization    ##############
    ##########################################################
    # Unpack options:
    [ 
        Add_Rest,
        Timeout,Return_Sol,
        Check_Small_Time,R_from_GITT,
        ] = Options
    ModelTimer = pb.Timer() # start counting time
    if Check_Small_Time == True:
        SmallTimer = pb.Timer()
    
    [BasicPath,Target] = Path_List
    # Gey output keys:
    keys_all = Get_Output_Keys(Para_dict)

    # Create folders
    if not os.path.exists(BasicPath +Target):
        os.mkdir(BasicPath +Target)
    if not os.path.exists(BasicPath +Target+"Mats"):
        os.mkdir(BasicPath +Target +"Mats");
    # Removed Plots and Excel folders creation if not needed, but keep Mats for data
    
    # define here:
    Run_i   = int(Para_dict["Run No"])  
    print(f'Start Now! Run {Run_i} Re {Re_No}')  
    Temp_K = Para_dict["Ageing temperature"]  

    # update 231205: write a function to get tot_cyc,cyc_age,update
    tot_cyc,cyc_age,update = Get_tot_cyc(Temp_K)
    Para_dict["Total ageing cycles"]       = int(tot_cyc)
    Para_dict["Ageing cycles between RPT"] = int(cyc_age)
    Para_dict["Update cycles for ageing"]  = int(update) # keys
    
    # set up experiment
    # update 231205: get a new function to initialize exp_AGE_text and exp_RPT_text
    V_max = 4.2     
    cap_0 = 4.86491   # set initial capacity to standardize SOH and get initial SEI thickness  
    V_min = 2.5; 
    [
        exp_AGE_text, step_AGE_CD, step_AGE_CC, step_AGE_CV,
        exp_breakin_text, exp_RPT_text, exp_RPT_GITT_text, 
        exp_refill,exp_adjust_before_age,
        step_0p1C_CD, step_0p1C_CC, step_0p1C_RE, step_0p5C_CD
        ] = Initialize_exp_text(
        V_max, V_min, Add_Rest)
    cycle_no = -1; 


    Sol_RPT = [];  Sol_AGE = [];   
    
    # pb.set_logging_level('INFO') # show more information!
    # set_start_method('fork') # from Patrick

    # Un-pack data:
    CyclePack,Para_0 = Para_init(Para_dict,4.86491) # initialize the parameter
    [
        Total_Cycles,Cycle_bt_RPT,Update_Cycles,
        RPT_Cycles,Temper_i,Temper_RPT,mesh_list,
        submesh_strech,model_options,
        cap_increase] = CyclePack
    [keys_all_RPT,keys_all_AGE] = keys_all
    str_exp_AGE_text  = str(exp_AGE_text)
    str_exp_RPT_text  = str(exp_RPT_text)
    str_exp_RPT_GITT_text  = str(exp_RPT_GITT_text)

    # define experiment
    Experiment_Long   = pb.Experiment( exp_AGE_text * Update_Cycles  )  
    # update 24-04-2023: delete GITT
    # Update 01-11-2023 add GITT back but with an option 
    # update 231210: refine experiment to avoid charge or hold at 4.2V
    if R_from_GITT: 
        Experiment_Breakin= pb.Experiment( 
            exp_breakin_text * 1
            + exp_RPT_GITT_text*24  + exp_refill * 1    # only do refil if have GITT
            + exp_adjust_before_age*1) 
        Experiment_RPT    = pb.Experiment( 
            exp_RPT_text * 1
            + exp_RPT_GITT_text*24  + exp_refill * 1    # only do refil if have GITT
            + exp_adjust_before_age*1) 
        Cyc_Index_Res = np.arange(1,25,1) 
    else:   # then get resistance from C/2
        Experiment_Breakin= pb.Experiment( 
            exp_breakin_text * 1
            + exp_adjust_before_age*1) 
        Experiment_RPT    = pb.Experiment( 
            exp_RPT_text * 1
            + exp_adjust_before_age*1) 
    

    #####  index definition ######################
    Small_Loop =  int(Cycle_bt_RPT/Update_Cycles);   
    SaveTimes = int(Total_Cycles/Cycle_bt_RPT);   

    # initialize my_dict for outputs
    my_dict_RPT = {}
    for keys in keys_all_RPT:
        for key in keys:
            my_dict_RPT[key]=[];
    my_dict_AGE = {}; 
    for keys in keys_all_AGE:
        for key in keys:
            my_dict_AGE[key]=[]
    my_dict_RPT["Cycle_RPT"] = []
    my_dict_RPT["Res_full"] = []
    my_dict_RPT["Res_midSOC"] = []
    my_dict_RPT["SOC_Res"] = []
    my_dict_AGE["Cycle_AGE"] = []
    my_dict_RPT["avg_Age_T"] = [] # Update add 230617 
    Cyc_Update_Index     =[]

    # update 220924: merge DryOut and Int_ElelyExces_Ratio
    ce_EC_0 = Para_0['EC initial concentration in electrolyte [mol.m-3]'] # used to calculate ce_EC_All
    
    Int_ElelyExces_Ratio = Para_0["Initial electrolyte excessive amount ratio"] ;
    DryOut = "On"
    print(f"Run {Run_i} Re {Re_No}: DryOut = {DryOut}")
    
    mdic_dry,Para_0 = Initialize_mdic_dry(Para_0,Int_ElelyExces_Ratio)
    
    if Check_Small_Time == True:
        print(f'Run {Run_i} Re {Re_No}: Spent {SmallTimer.time()} on Initialization')
        SmallTimer.reset()
    ##########################################################
    ##############    Part-2: Run model         ##############
    ##########################################################
    ##########################################################
    Timeout_text = 'I timed out'
    ##########    2-1: Define model and run break-in cycle
    solver_type = Para_dict.get("Solver_Type", "Casadi")
    try:  
        # Timelimit = int(3600*2)
        # the following turns on for HPC only!
        if Timeout == True:
            timeout_RPT = TimeoutFunc(
                Run_Breakin, 
                timeout=Timelimit, 
                timeout_val=Timeout_text)
            Result_list_breakin  = timeout_RPT(
                model_options, Experiment_Breakin, 
                Para_0, mesh_list, submesh_strech,
                cap_increase,
                solver_type=solver_type)
        else:
            Result_list_breakin  = Run_Breakin(
                model_options, Experiment_Breakin, 
                Para_0, mesh_list, submesh_strech,
                cap_increase,
                solver_type=solver_type)
        [Model_0,Sol_0,Call_Breakin] = Result_list_breakin
        if Return_Sol == True:
            Sol_RPT.append(Sol_0)
        if Call_Breakin.success == False:
            print("Fail due to Experiment error or infeasible")
            1/0
        if Sol_0 == Timeout_text: # to do: distinguish different failure cases
            print("Fail due to Timeout")
            1/0
        if Sol_0 == "Model error or solver error":
            print("Fail due to Model error or solver error")
            1/0
    except ZeroDivisionError as e:
        str_error_Breakin = str(e)
        if Check_Small_Time == True:
            str_error_Breakin = f"Run {Run_i} Re {Re_No}: Fail break-in cycle within {SmallTimer.time()}, need to exit the whole scan now due to {str_error_Breakin} but do not know how!"
            print(str_error_Breakin)
            SmallTimer.reset()
        else:
            str_error_Breakin = f"Run {Run_i} Re {Re_No}: Fail break-in cycle, need to exit the whole scan now due to {str_error_Breakin} but do not know how!"
            print(str_error_Breakin)
        Flag_Breakin = False 
    else:
        if Check_Small_Time == True:    
            print(f"Run {Run_i} Re {Re_No}: Finish break-in cycle within {SmallTimer.time()}")
            SmallTimer.reset()
        else:
            print(f"Run {Run_i} Re {Re_No}: Finish break-in cycle")
        # post-process for break-in cycle - 0.1C only
        my_dict_RPT = GetSol_dict (my_dict_RPT,keys_all_RPT, Sol_0, 
            0, step_0p1C_CD, step_0p1C_CC,step_0p1C_RE , step_AGE_CV   )
        # update 230517 - Get R from C/2 discharge only, discard GITT
        cap_full = 5; 
        if R_from_GITT: 
            Res_midSOC,Res_full,SOC_Res = Get_0p1s_R0(Sol_0,Cyc_Index_Res,cap_full)
        else: 
            step_0P5C_CD = Sol_0.cycles[0].steps[step_0p5C_CD]
            Res_midSOC,Res_full,SOC_Res = Get_R_from_0P5C_CD(step_0P5C_CD,cap_full)
        my_dict_RPT["SOC_Res"].append(SOC_Res)
        my_dict_RPT["Res_full"].append(Res_full)
        my_dict_RPT["Res_midSOC"].append(Res_midSOC)    

        my_dict_RPT["avg_Age_T"].append(Temper_i-273.15)  # Update add 230617              
        del SOC_Res,Res_full,Res_midSOC
        cycle_count =0
        my_dict_RPT["Cycle_RPT"].append(cycle_count)
        Cyc_Update_Index.append(cycle_count)
        Flag_Breakin = True
        if Check_Small_Time == True:    
            print(f"Run {Run_i} Re {Re_No}: Finish post-process for break-in cycle within {SmallTimer.time()}")
            SmallTimer.reset()
        else:
            print(f"Run {Run_i} Re {Re_No}: Finish post-process for break-in cycle")
        
    Flag_AGE = True; Flag_partial_AGE = False
    str_error_AGE_final = "Empty";   str_error_RPT = "Empty"; 
    DeBug_List_RPT = "Break in fail"; DeBug_List_AGE = "Break in fail"
    #############################################################
    #######   2-2: Write a big loop to finish the long experiment    
    if Flag_Breakin == True: 
        k=0
        # Para_All.append(Para_0);Model_All.append(Model_0);Sol_All_i.append(Sol_0); 
        Para_0_Dry_old = Para_0;     Model_Dry_old = Model_0  ; Sol_Dry_old = Sol_0;   del Model_0,Sol_0
        while k < SaveTimes:    
            i=0  
            avg_Age_T = []  
            while i < Small_Loop:
                Data_Pack,Paraupdate   = Cal_new_con_Update (  Sol_Dry_old,   Para_0_Dry_old )
                # Run aging cycle:
                try:
                    #Timelimit = int(60*60*2)
                    if Timeout == True:
                        timeout_AGE = TimeoutFunc(
                            Run_Model_Base_On_Last_Solution, 
                            timeout=Timelimit, 
                            timeout_val=Timeout_text)
                        Result_list_AGE = timeout_AGE( 
                            Model_Dry_old  , Sol_Dry_old , Paraupdate ,Experiment_Long, 
                            Update_Cycles,Temper_i,mesh_list,submesh_strech, solver_type=solver_type )
                    else:
                        Result_list_AGE = Run_Model_Base_On_Last_Solution( 
                            Model_Dry_old  , Sol_Dry_old , Paraupdate ,Experiment_Long, 
                            Update_Cycles,Temper_i,mesh_list,submesh_strech, solver_type=solver_type )
                    [Model_Dry_i, Sol_Dry_i , Call_Age,DeBug_List_AGE ] = Result_list_AGE
                    
                    if Return_Sol == True:
                        Sol_AGE.append(Sol_Dry_i)
                    if "Partially" in DeBug_List_AGE[-1]:
                        Flag_partial_AGE = True
                        succeed_cycs = len(Sol_Dry_i.cycles) 
                        if succeed_cycs < Update_Cycles:
                            print(f"Instead of {Update_Cycles}, succeed only {succeed_cycs} cycles")
                            Flag_partial_AGE = True
                    else:
                        if Call_Age.success == False:
                            print("Fail due to Experiment error or infeasible")
                            str_error_AGE = "Experiment error or infeasible"
                            1/0
                        elif Sol_Dry_i == "Model error or solver error":
                            print("Fail due to Model error or solver error")
                            str_error_AGE = "Model error or solver error"
                            1/0
                        else:
                            pass
                    if Sol_Dry_i == Timeout_text: # fail due to timeout
                        print("Fail due to Timeout")
                        str_error_AGE = "Timeout"
                        1/0
                except ZeroDivisionError as e: # ageing cycle fails
                    if Check_Small_Time == True:    
                        str_error_AGE_final = f"Run {Run_i} Re {Re_No}: Fail during No.{Cyc_Update_Index[-1]} ageing cycles within {SmallTimer.time()} due to {str_error_AGE}"
                        print(str_error_AGE_final)
                        SmallTimer.reset()
                    else:
                        str_error_AGE_final = f"Run {Run_i} Re {Re_No}: Fail during No.{Cyc_Update_Index[-1]} ageing cycles due to {str_error_AGE}"
                        print(str_error_AGE_final)
                    Flag_AGE = False
                    break
                else:                           # ageing cycle SUCCEED
                    succeed_cycs = len(Sol_Dry_i.cycles) 
                    Para_0_Dry_old = Paraupdate; Model_Dry_old = Model_Dry_i; Sol_Dry_old = Sol_Dry_i;   
                    del Paraupdate,Model_Dry_i,Sol_Dry_i
                    
                    if Check_Small_Time == True:    
                        print(f"Run {Run_i} Re {Re_No}: Finish for No.{Cyc_Update_Index[-1]} ageing cycles within {SmallTimer.time()}")
                        SmallTimer.reset()
                    else:
                        print(f"Run {Run_i} Re {Re_No}: Finish for No.{Cyc_Update_Index[-1]} ageing cycles")

                    # post-process for first ageing cycle and every -1 ageing cycle
                    if k==0 and i==0:    
                        my_dict_AGE = GetSol_dict (my_dict_AGE,keys_all_AGE, Sol_Dry_old, 
                            0, step_AGE_CD , step_AGE_CC , step_0p1C_RE, step_AGE_CV   )     
                        my_dict_AGE["Cycle_AGE"].append(1)
                    # update 240111
                    if Flag_partial_AGE == True:
                        try:
                            my_dict_AGE = GetSol_dict (my_dict_AGE,keys_all_AGE, Sol_Dry_old, 
                                cycle_no, step_AGE_CD , step_AGE_CC , step_0p1C_RE, step_AGE_CV   )    
                        except IndexError:
                            print("The last cycle is incomplete, try [-2] cycle")
                            try:
                                my_dict_AGE = GetSol_dict (my_dict_AGE,keys_all_AGE, Sol_Dry_old, 
                                    -2, step_AGE_CD , step_AGE_CC , step_0p1C_RE, step_AGE_CV   )   
                            except IndexError:
                                print("[-2] cycle also does not work, try first one")
                                try:
                                    my_dict_AGE = GetSol_dict (my_dict_AGE,keys_all_AGE, Sol_Dry_old, 
                                        0, step_AGE_CD , step_AGE_CC , step_0p1C_RE, step_AGE_CV   )  
                                except:
                                    print("Still does not work, less than one cycle, we are in trouble")
                    else:
                        try:
                            my_dict_AGE = GetSol_dict (my_dict_AGE,keys_all_AGE, Sol_Dry_old, 
                                cycle_no, step_AGE_CD , step_AGE_CC , step_0p1C_RE, step_AGE_CV   )    
                        except:
                            print("GetSol_dict fail for a complete ageing set for unknown reasons!!!")
                    cycle_count +=  succeed_cycs 
                    avg_Age_T.append(np.mean(
                        Sol_Dry_old["Volume-averaged cell temperature [C]"].entries))
                    my_dict_AGE["Cycle_AGE"].append(cycle_count)           
                    Cyc_Update_Index.append(cycle_count)
                    
                    mdic_dry = Update_mdic_dry(Data_Pack,mdic_dry)
                    
                    if Check_Small_Time == True:    
                        print(f"Run {Run_i} Re {Re_No}: Finish post-process for No.{Cyc_Update_Index[-1]} ageing cycles within {SmallTimer.time()}")
                        SmallTimer.reset()
                    else:
                        pass
                    i += 1;   ##################### Finish small loop and add 1 to i 
            
            # run RPT, and also update parameters (otherwise will have problems)
            Data_Pack , Paraupdate  = Cal_new_con_Update (  
                Sol_Dry_old,   Para_0_Dry_old   )
            try:
                # Timelimit = int(60*60*2)
                if Timeout == True:
                    timeout_RPT = TimeoutFunc(
                        Run_Model_Base_On_Last_Solution_RPT, 
                        timeout=Timelimit, 
                        timeout_val=Timeout_text)
                    Result_list_RPT = timeout_RPT(
                        Model_Dry_old  , Sol_Dry_old ,   
                        Paraupdate,      Experiment_RPT, RPT_Cycles, 
                        Temper_RPT ,mesh_list ,submesh_strech, solver_type=solver_type 
                    )
                else:
                    Result_list_RPT = Run_Model_Base_On_Last_Solution_RPT(
                        Model_Dry_old  , Sol_Dry_old ,   
                        Paraupdate,      Experiment_RPT, RPT_Cycles, 
                        Temper_RPT ,mesh_list ,submesh_strech, solver_type=solver_type 
                    )
                [Model_Dry_i, Sol_Dry_i,Call_RPT,DeBug_List_RPT]  = Result_list_RPT
                if Return_Sol == True:
                    Sol_RPT.append(Sol_Dry_i)
                #print(f"Temperature for RPT is now: {Temper_RPT}")  
                if Call_RPT.success == False:
                    print("Fail due to Experiment error or infeasible")
                    str_error_RPT = "Experiment error or infeasible"
                    1/0 
                if Sol_Dry_i == Timeout_text:
                    #print("Fail due to Timeout")
                    str_error_RPT = "Timeout"
                    1/0
                if Sol_Dry_i == "Model error or solver error":
                    print("Fail due to Model error or solver error")
                    str_error_RPT = "Model error or solver error"
                    1/0
            except ZeroDivisionError as e:
                if Check_Small_Time == True:    
                    str_error_RPT = f"Run {Run_i} Re {Re_No}: Fail during No.{Cyc_Update_Index[-1]} RPT cycles within {SmallTimer.time()}, due to {str_error_RPT}"
                    print(str_error_RPT)
                    SmallTimer.reset()
                else:
                    str_error_RPT = f"Run {Run_i} Re {Re_No}: Fail during No.{Cyc_Update_Index[-1]} RPT cycles, due to {str_error_RPT}"
                    print(str_error_RPT)
                break
            else:
                # post-process for RPT
                Cyc_Update_Index.append(cycle_count)
                if Check_Small_Time == True:    
                    print(f"Run {Run_i} Re {Re_No}: Finish for No.{Cyc_Update_Index[-1]} RPT cycles within {SmallTimer.time()}")
                    SmallTimer.reset()
                else:
                    print(f"Run {Run_i} Re {Re_No}: Finish for No.{Cyc_Update_Index[-1]} RPT cycles")
                # update 231210: delete the first hold at 4.2V for later RPT
                my_dict_RPT = GetSol_dict (my_dict_RPT,keys_all_RPT, Sol_Dry_i, 
                    0,step_0p1C_CD, step_0p1C_CC,step_0p1C_RE , step_AGE_CV   ) 
                my_dict_RPT["Cycle_RPT"].append(cycle_count)
                my_dict_RPT["avg_Age_T"].append(np.mean(avg_Age_T))  # Make sure avg_Age_T and 
                
                # update 230517 - Get R from C/2 discharge only, discard GITT
                cap_full = Paraupdate["Nominal cell capacity [A.h]"] # 5
                if R_from_GITT: 
                    Res_midSOC,Res_full,SOC_Res = Get_0p1s_R0(Sol_Dry_i,Cyc_Index_Res,cap_full)
                else: 
                    step_0P5C_CD = Sol_Dry_i.cycles[0].steps[step_0p5C_CD]
                    Res_midSOC,Res_full,SOC_Res = Get_R_from_0P5C_CD(step_0P5C_CD,cap_full)
                my_dict_RPT["SOC_Res"].append(SOC_Res)
                my_dict_RPT["Res_full"].append(Res_full)
                my_dict_RPT["Res_midSOC"].append(Res_midSOC)             
                del SOC_Res,Res_full,Res_midSOC
                mdic_dry = Update_mdic_dry(Data_Pack,mdic_dry)
                Para_0_Dry_old = Paraupdate;    Model_Dry_old = Model_Dry_i  ;     Sol_Dry_old = Sol_Dry_i    ;   
                del Paraupdate,Model_Dry_i,Sol_Dry_i
                if Check_Small_Time == True:    
                    print(f"Run {Run_i} Re {Re_No}: Finish post-process for No.{Cyc_Update_Index[-1]} RPT cycles within {SmallTimer.time()}")
                    SmallTimer.reset()
                else:
                    pass
                if Flag_AGE == False or Flag_partial_AGE == True:
                    break
            k += 1 
    DeBug_Lists = [DeBug_List_RPT,DeBug_List_AGE]
    Keys_error = ["Error tot %","Error SOH %","Error LLI %",
        "Error LAM NE %","Error LAM PE %",
        "Error Res %","Error ageT %","Punish"]
    ############################################################# 
    #########   An extremely bad case: cannot even finish breakin
    if Flag_Breakin == False: 
        midc_merge = [my_dict_RPT, my_dict_AGE,mdic_dry]
        import pickle
        with open(
            BasicPath + Target+"Mats/" 
            + str(Run_i)+ f'-DeBug_Lists_Re_{Re_No}.pkl', 'wb') as file:
            pickle.dump(DeBug_Lists, file)

        return midc_merge,Sol_RPT,Sol_AGE,DeBug_Lists
    ##########################################################
    ##############   Part-3: Post-prosessing    ##############
    ##########################################################
    # Simplified: Save data only
    else:
        # Update 230221 - Add model LLI, LAM manually 
        my_dict_RPT = Get_SOH_LLI_LAM(my_dict_RPT,model_options,"On",mdic_dry,cap_0)

        if Check_Small_Time == True:    
            print(f"Run {Run_i} Re {Re_No}: Getting extra variables within {SmallTimer.time()}")
            SmallTimer.reset()
        else:
            pass
        
        # Removed comparison with experimental data
        Pass_Fail = "Pass"

        ##########################################################
        #########      3-1: Plot cycle,location, Dryout related 
        # update 23-05-25 there is a bug in Cyc_Update_Index, need to slide a bit:
        Cyc_Update_Index.insert(0,0); del Cyc_Update_Index[-1]
        
        # Removed Plotting calls

        ##########################################################
        ##########################################################
        #########      3-3: Save summary to excel (Removed)
        # Removed Write_Dict_to_Excel
        
        # pick only the cyc - most concern
        Keys_cyc_mat =[
            'Throughput capacity [kA.h]',
            'CDend SOH [%]',
            "CDend LLI [%]",
            "CDend LAM_ne [%]",
            "CDend LAM_pe [%]",
            "Res_midSOC",
        ]
        my_dict_mat = {}
        for key in Keys_cyc_mat:
            my_dict_mat[key]=my_dict_RPT[key]
        #########      3-2: Save data as .mat or .json
        my_dict_RPT["Cyc_Update_Index"] = Cyc_Update_Index
        my_dict_RPT["SaveTimes"]    = SaveTimes
        # new_dict = {}
        # add "age" after age keys to avoid overwritten
        #for key in my_dict_AGE: 
        #    new_key = key + " Age"
        #    new_dict[new_key] = my_dict_AGE[key]
        # midc_merge = {**my_dict_RPT, **my_dict_AGE,**mdic_dry}
        midc_merge = [my_dict_RPT, my_dict_AGE,mdic_dry]
        if isinstance(Sol_RPT[-1],pb.solvers.solution.Solution):
            _,dict_short = Get_Last_state(Model_Dry_old, Sol_RPT[-1])
            sol_last = Sol_RPT[-1]
        else:
            _,dict_short = Get_Last_state(Model_Dry_old, Sol_AGE[-1])
            sol_last = Sol_AGE[-1]
            print("!!!!!!!Big problem! The last RPT fails, need to restart from RPT next time")
        # calculate dry-out parameter first
        Data_Pack,Paraupdate   = Cal_new_con_Update (  sol_last,   Para_0_Dry_old )
        i_try = 0
        while i_try<3:
            try:
                getSth = sol_last['Throughput capacity [A.h]'].entries[-1] 
                # Sol_new['Throughput capacity [A.h]'].entries[-1]
            except:
                i_try += 1
                print(f"Fail to read Throughput capacity for the {i_try}th time")
            else:
                break
        Save_for_Reload = [ midc_merge, dict_short, Paraupdate, Data_Pack, getSth]
        import pickle,json

        with open(
            BasicPath + Target+"Mats/" 
            + str(Run_i)+ f'_Re_{Re_No}-midc_merge.pkl', 'wb') as file:
            pickle.dump(midc_merge, file)
        
        with open(
            BasicPath + Target+"Mats/" 
            + str(Run_i)+ f'_Re_{Re_No}-Save_for_Reload.pkl', 'wb') as file:
            pickle.dump(Save_for_Reload, file)


        try:
            savemat(
                BasicPath + Target+"Mats/" 
                + str(Run_i)+ f'_Re_{Re_No}-Ageing_summary_only.mat',
                my_dict_mat)  
        except:
            print(f"Run {Run_i} Re {Re_No}: Encounter problems when saving mat file!")
        else: 
            print(f"Run {Run_i} Re {Re_No}: Successfully save mat file!")

        if Check_Small_Time == True:    
            print(f"Run {Run_i} Re {Re_No}: Try saving within {SmallTimer.time()}")
            SmallTimer.reset()
        else:
            pass

        
        with open(
            BasicPath + Target+"Mats/" 
            + str(Run_i)+ f'_Re_{Re_No}-DeBug_Lists.pkl', 'wb') as file:
            pickle.dump(DeBug_Lists, file)
        

        # update 231217: save ageing solution if partially succeed in ageing set
        if Flag_partial_AGE == True:
            try:
                Sol_partial_AGE_list = [
                    Sol_AGE[-1].cycles[0],    Sol_AGE[-1].cycles[-1],
                    len(Sol_AGE[-1].cycles)]
            except IndexError:
                try:
                    Sol_partial_AGE_list = [
                        Sol_AGE[-1].cycles[0],    Sol_AGE[-1].cycles[-2],
                        len(Sol_AGE[-1].cycles)]
                except IndexError:
                    print("Problems in saving last two cycles in Sol_AGE[-1],"
                            " now saving the whole Sol_AGE[-1]")
                    Sol_partial_AGE_list = [
                        Sol_AGE[-1],    "nan",
                        len(Sol_AGE[-1].cycles)]
            with open(
                BasicPath + Target+"Mats/" 
                + str(Run_i)+ f'_Re_{Re_No}-Sol_partial_AGE_list.pkl', 'wb') as file:
                pickle.dump(Sol_partial_AGE_list, file)
            print(f"Last AGE succeed partially, save Sol_partial_AGE_list.pkl for Run {Run_i} Re {Re_No}")
        else:
            pass
        print("Succeed doing something in {}".format(ModelTimer.time()))
        print(f'This is the end of No. {Run_i} scan, Re {Re_No}')
        return midc_merge,Sol_RPT,Sol_AGE,DeBug_Lists


