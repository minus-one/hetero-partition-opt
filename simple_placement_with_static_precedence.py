"""
Optimizer to study placement given a total ordering of tasks 
in a multi device scenario for a set of tasks.
Precedence is defined between tasks explicitly and statically used to synthesize the constraints.
The topological ordering is followed.
Each device has to execute one task after the other.
"""
import math
import argparse
from ortools.linear_solver import pywraplp

import graph_utils
import tensor_info as T
import node as N

LARGE_NUM=1000
PCIE_BW=32.0*1E3
NUM_THREADS=8

def setup_graph_info(args):
    graph_utils.process_node_file(args.graph_file)
    graph_utils.process_runtime_file(args.gpu_runtime_file, key_name="seq_gpu")
    graph_utils.process_tensor_file(args.tensor_info_file)
    graph_utils.process_runtime_file(args.cpu_runtime_file, key_name="seq_cpu")

    # Prune from the graph mem-copies and match predecessor with successor
    task_list = []
    service_times = {'cpu' : {}, 'gpu' : {}}
    pred_info = {}
    order_info = {} # Maps node-idx to the position in the topological order
    device_list = ['cpu', 'gpu']
    #device_list = ['gpu']

    memcpy_node_idx = []
    for node_idx in graph_utils.G.nodes():
        if "Memcpy" in N.Node.all_nodes[node_idx].node_type:
            # Prune
            for pred_node_idx in graph_utils.get_node_predecessors(node_idx):
                for succ_node_idx in graph_utils.get_node_successors(node_idx):
                    graph_utils.G.add_edge(pred_node_idx, succ_node_idx)
                    T.TensorInfo.get_tensor_from_dep_info(pred_node_idx, node_idx).add_node_dep(succ_node_idx)
            memcpy_node_idx.append(node_idx)
    graph_utils.G.remove_nodes_from(memcpy_node_idx)
    
    # Uncomment for pruning the number of nodes in the graph and running for a smaller problem size
    #max_num_nodes = 500
    #print("Pruning input to: ", max_num_nodes, " nodes")
    #node_idx_list = []
    #count = 0
    #for node_idx in graph_utils.G.nodes():
    #    node_idx_list.append(node_idx)
    #    count += 1
    #    if count >= max_num_nodes:
    #        break;

    #node_rem_list = []
    #for node_idx in graph_utils.G.nodes():
    #    if node_idx not in node_idx_list:
    #        node_rem_list.append(node_idx)
    #graph_utils.G.remove_nodes_from(node_rem_list)

    for (topo_idx,node_idx) in enumerate(graph_utils.get_topo_order_of_nodes(graph_utils.G)):
        task_list.append(node_idx)
        service_times['cpu'][node_idx] = int(N.Node.all_nodes[node_idx].run_time['seq_cpu'].run_time)
        service_times['gpu'][node_idx] = int(N.Node.all_nodes[node_idx].run_time['seq_gpu'].run_time)
        order_info[node_idx]  = topo_idx
        
        pred_info[node_idx] = []
        for pred_node_idx in graph_utils.get_node_predecessors(node_idx):
            dep_tensor = T.TensorInfo.get_tensor_from_dep_info(pred_node_idx, node_idx) 
            if dep_tensor:
                mem_size = dep_tensor.size
            else:
                print("Tensor missing, Node: ", node_idx, " PredNode: ", pred_node_idx)
            pred_info[node_idx].append((pred_node_idx, mem_size))

    #print(task_list)
    #print(device_list)
    #print(service_times)
    #print(pred_info)
    return task_list, device_list, service_times, pred_info, order_info

def get_dummy_info():
    global PCIE_BW
    device_list = ['cpu', 'gpu']
    task_list = [0, 1, 2, 3, 4]
    service_times = {'cpu' : {0:2, 1:2, 2:1, 3:2, 4:3}, 'gpu': {0:2, 1:3, 2:4, 3:2, 4:3}}
    # Map of node to its predecessors and the data transfer incurred for that predecessor
    #pred_info = {0: [], 1: [(0,1)], 2: [(1,1)], 3: [], 4: []}
    pred_info = {0: [], 1: [(0, 2)], 2: [(1, 3)], 3: [], 4: []}
    order_info = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    PCIE_BW = 1.0
    return task_list, device_list, service_times, pred_info, order_info
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_file", help="File containing the graph info", default="ort_graph_dump_file.csv")
    parser.add_argument("--gpu_runtime_file", help="File containing the runtime info", default="ort_seq_mean.csv")
    parser.add_argument("--cpu_runtime_file", help="File containing the CPU runtime info", default="ort_seq_cpu_mean.csv")
    parser.add_argument("--tensor_info_file", help="File containing the tensor info", default="ort_node_memory_dump_file.csv")
    parser.add_argument("--dummy_run", help="Perform a dummy run with small inputs", action='store_true', default=False)
    parser.add_argument("--use_gurobi", help="Use gurobi as backend", action='store_true', default=False)
    parser.add_argument("--dump_lp", help="Dump the LP generated (does not work with large LPs)", action='store_true', default=False)
    args = parser.parse_args()

    if args.dummy_run:
        task_list, device_list, service_times, pred_info, order_info = get_dummy_info()
    else:
        task_list, device_list, service_times, pred_info, order_info = setup_graph_info(args)

    print("Num Nodes: ", len(task_list))
    print("Devices: ", device_list)
    for device, device_service_times in service_times.items():
        print("Device: ", device, " Num Nodes: ", len(device_service_times.keys()))
    print("PredInfo: ", len(pred_info.keys()))
    print("Order Info: ", len(order_info.keys()))
    optimize_placement_and_ordering(args, task_list, device_list, service_times, pred_info, order_info)
    
def optimize_placement_and_ordering(args, task_list, device_list, service_times, pred_info, order_info):
    global LARGE_NUM
    # Sanity check
    assert(len(device_list) <= len(service_times.keys()))
    for device, device_service_times in service_times.items():
        assert(len(task_list) <= len(device_service_times.keys()))
    assert(len(task_list) <= len(pred_info.keys()))
    
    # Set bignum based on numbers provided
    max_make_span = 0.0
    for (device, st_map) in service_times.items():
        max_device_make_span = 0.0
        for (_, st) in st_map.items():
            max_device_make_span += st
        max_make_span = max(max_make_span, max_device_make_span)

    LARGE_NUM = max(LARGE_NUM, max_make_span * 10)
    print("Setting a Large enough BigNum: ", LARGE_NUM)

    if args.use_gurobi:
        solver = pywraplp.Solver('task_scheduler_mip', pywraplp.Solver.GUROBI_MIXED_INTEGER_PROGRAMMING)
    else:
        solver = pywraplp.Solver('task_scheduler_mip', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    inf = solver.infinity()
    
    exec_time = solver.IntVar(0.0, inf, 'ExecTime')
    
    start_times = {}
    finish_times = {}
    comm_delay = {}
    placement = {}
    for d in device_list:
        start_times[d] = {}
        finish_times[d] = {}
        comm_delay[d] = {}
        placement[d] = {}
        for i in task_list:
            start_times[d][i] = solver.IntVar(0.0, inf, 'StartTime_' + str(d) + '_' + str(i))
            finish_times[d][i] = solver.IntVar(0.0, inf, 'FinishTime_' + str(d) + '_' + str(i))
            comm_delay[d][i] = solver.IntVar(0.0, inf, 'CommDelay_' + str(d) + '_' + str(i))
            placement[d][i] = solver.IntVar(0.0, 1.0, 'D_' + str(d) + '_' + str(i))

    ordering = {}
    for i in task_list:
        ordering[i] = {}
        for j in task_list:
            if i == j:
                ordering[i][j] = 0
            else:
                if order_info[i] < order_info[j]:
                    ordering[i][j] = 1
                else:
                    ordering[i][j] = 0

    print('Number of variables=', solver.NumVariables())
    
    # ExecTime - FinishTime_d_i >= 0
    for d in device_list:
        for i in task_list:
            constr = solver.Constraint(0.0, inf)
            constr.SetCoefficient(exec_time, 1)
            constr.SetCoefficient(finish_times[d][i], -1)
    
    # Placement constraints
    for i in task_list:
        constr = solver.Constraint(1.0, 1.0)
        for d in device_list:
            constr.SetCoefficient(placement[d][i], 1)

    # Communication Delay
    for d1 in device_list:
        for i in task_list:
            # CommDelay_d_i - sigma(D_l_j*(mem_size/BW)) + (1-D_d_i)*H >= 0 (l not= d)
            # CommDelay_d_i = 0 if d is not the device on which it is executing, 
            # else it will calculate the CommDelay for the device
            constr1 = solver.Constraint(-LARGE_NUM, inf)
            constr1.SetCoefficient(comm_delay[d1][i], 1)
            for (j, mem_size) in pred_info[i]:
                for d2 in device_list:
                    if d1 == d2:
                        continue
                    constr1.SetCoefficient(placement[d2][j], (-1)*int(math.ceil(mem_size/PCIE_BW)))
            constr1.SetCoefficient(placement[d1][i], (-1)*LARGE_NUM)
            
            constr2a = solver.Constraint(0.0, inf)
            constr2a.SetCoefficient(comm_delay[d1][i], 1)
            constr2a.SetCoefficient(placement[d1][i], (+1)*LARGE_NUM)
            constr2b = solver.Constraint(0.0, inf)
            constr2b.SetCoefficient(comm_delay[d1][i], -1)
            constr2b.SetCoefficient(placement[d1][i], (+1)*LARGE_NUM)

    # Finish Time calculation
    for d in device_list:
        for i in task_list:
            # FinishTime_d_i - StartTime_d_i - (D_d_i * ServiceTime_d_i) - CommDelay_d_i = 0
            constr = solver.Constraint(0.0, 0.0)
            constr.SetCoefficient(finish_times[d][i], 1)
            constr.SetCoefficient(start_times[d][i], -1)
            constr.SetCoefficient(placement[d][i], (-1) * service_times[d][i])
            constr.SetCoefficient(comm_delay[d][i], -1)
    
    # If not executing on particular device, set StartTime = 0, else set StartTime >= 0
    # -(D_d_i * H) <= StartTime_d_i <= (D_d_i * H) ==> StartTime_d_i + (D_d_i * H) >= 0 && -StartTime_d_i + D_d_i *H >= 0
    # StartTime_d_i + (1-D_d_i) * H >= 0
    for d in device_list:
        for i in task_list:
            constr1a = solver.Constraint(0.0, inf)
            constr1a.SetCoefficient(start_times[d][i], 1)
            constr1a.SetCoefficient(placement[d][i], LARGE_NUM)
            constr1b = solver.Constraint(0.0, inf)
            constr1b.SetCoefficient(start_times[d][i], -1)
            constr1b.SetCoefficient(placement[d][i], LARGE_NUM)

            constr2 = solver.Constraint(-LARGE_NUM, inf)
            constr2.SetCoefficient(start_times[d][i], 1)
            constr2.SetCoefficient(placement[d][i], (-1) * LARGE_NUM)

    # Exclusion constraints
    for d in device_list:
        for i in task_list:
            for j in task_list:
                if i == j:
                    continue
                if ordering[i][j] == 1:
                    # This means StartTime_d_j must be completely after i has completed execution on this device, if indeed j is assigned to this device d
                    # StartTime_d_j - StartTime_d_i + (1 - D_d_j) * H >= 0
                    constr1 = solver.Constraint(-LARGE_NUM, inf)
                    #constr1 = solver.Constraint(0.0, inf)
                    constr1.SetCoefficient(start_times[d][j], 1)
                    constr1.SetCoefficient(start_times[d][i], -1)
                    constr1.SetCoefficient(placement[d][j], (-1) * LARGE_NUM)
                    # StartTime_d_j - FinishTime_d_i + (1 - D_d_j) * H >= 0
                    constr2 = solver.Constraint(-LARGE_NUM, inf)
                    #constr2 = solver.Constraint(0.0, inf)
                    constr2.SetCoefficient(start_times[d][j], 1)
                    constr2.SetCoefficient(finish_times[d][i], -1)
                    constr2.SetCoefficient(placement[d][j], (-1) * LARGE_NUM)
                #elif ordering[i][j] == 0:
                #    # This means i must execute after j completely
                #    # StartTime_d_i - StartTime_d_j >= 0
                #    constr1 = solver.Constraint(0.0, inf)
                #    constr1.SetCoefficient(start_times[d][i], 1)
                #    constr1.SetCoefficient(start_times[d][j], -1)
                #    # StartTime_d_i - FinishTime_d_j >= 0
                #    constr2 = solver.Constraint(0.0, inf)
                #    constr2.SetCoefficient(start_times[d][i], 1)
                #    constr2.SetCoefficient(finish_times[d][j], -1)

    precedence_list = []
    # Precedence constraints
    for succ in pred_info.keys():
        for (pred, _) in pred_info[succ]:
            precedence_list.append((pred, succ))
    for (pred, succ) in precedence_list:
        # StartTime_d1_succ - FinishTime_d2_pred + (1 - D_d1_succ)*H >= 0
        for d1 in device_list:
            for d2 in device_list:
                time_constr = solver.Constraint(-LARGE_NUM, inf)
                time_constr.SetCoefficient(start_times[d1][succ], 1) 
                time_constr.SetCoefficient(finish_times[d2][pred], -1)
                time_constr.SetCoefficient(placement[d1][succ], (-1) * LARGE_NUM)

    print("Number of constraints= ", solver.NumConstraints())

    obj1 = solver.Objective()
    obj1.SetMinimization()
    obj1.SetCoefficient(exec_time, 1)
   
    if args.dump_lp:
        with open("LP.txt", "w") as lp_file:
            #lp_file.write(solver.ExportModelAsLpFormat(False).replace('\\', '').replace(',_', ','))
            lp_file.write(solver.ExportModelAsLpFormat(False))
            lp_file.close()
        print("LP Written to file...")
    print("Starting solver...")
   
    out_file = open("opt_output.txt", "w")

    #solver.SetNumThreads(NUM_THREADS)
    solver.SetTimeLimit(60 * 60 * 24 * 1000) # 1day
    start_time = solver.WallTime()
    status = solver.Solve()
    end_time = solver.WallTime()
    out_file.write("Solver took: " + str(end_time - start_time) + " ms\n")

    def write_solution(solver, out_file):
        out_file.write("===============Solution==============\n")
        out_file.write("Obj.: ExecTime = " + str(solver.Objective().Value()) + "\n")

        out_file.write("Task | Device | StartTime | ServiceTime | CommDelay | FinishTime\n")
        for i in task_list:
            for d in device_list:
                if placement[d][i].solution_value() == 0.0:
                    continue
                out_file.write(str(i) + " | " + str(d) + " | " + str(start_times[d][i].solution_value()) + " | " \
                        + str(service_times[d][i]) + " | " + str(comm_delay[d][i].solution_value()) + " | "\
                        + str(finish_times[d][i].solution_value()) + "\n")
        
        out_file.write("------Raw numbers------\n")
        out_file.write("Placement:\n")
        for d in device_list:
            for i in task_list:
                out_file.write(str(placement[d][i].solution_value()) + " ")
            out_file.write("\n")

        out_file.write("Task | Device | StartTime | ServiceTime | CommDelay | FinishTime\n")
        for i in task_list:
            for d in device_list:
                out_file.write(str(i) + " | " + str(d) + " | " + str(start_times[d][i].solution_value()) + " | " \
                        + str(service_times[d][i]) + " | " + str(comm_delay[d][i].solution_value()) + " | "\
                        + str(finish_times[d][i].solution_value()) + "\n")

        out_file.write("Ordering:\n")
        for i in task_list:
            for j in task_list:
                out_file.write(str(ordering[i][j]) + ", ")
            out_file.write("\n")

    if status == pywraplp.Solver.OPTIMAL:
        print("Opt found")
        write_solution(solver, out_file)

        while(solver.NextSolution()):
            write_solution(solver, out_file)
        else:
            out_file.write("No other solutions exist...\n")
     
    else:
        print("No Opt")
        if status == pywraplp.Solver.FEASIBLE:
            print("Showing a feasible solution")
            write_solution(solver, out_file)
        else:
            print("No feasible solution")
            out_file.write("The problem does not have any solution\n")

    out_file.close()
    print("Opt output written...")

if __name__ == '__main__':
    main()
