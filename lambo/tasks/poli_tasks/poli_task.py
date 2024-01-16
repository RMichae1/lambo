from multiprocessing import pool
import numpy as np
from pathlib import Path

from poli import objective_factory
from poli.core.util.external_observer import ExternalObserver
from poli.core.util.proteins.mutations import find_closest_wildtype_pdb_file_to_mutant

from lambo.candidate import StringCandidate
from lambo.tasks.base_task import BaseTask
from lambo.tasks.poli_tasks import ALGORITHM, STARTING_N, BATCH_SIZE
from lambo.tasks.poli_tasks import POLI_TASK_HYDRA_KEY
from lambo.tasks.poli_tasks import TRACKING_URI

from corel.observers.poli_base_logger import PoliBaseMlFlowObserver


global problem_information, f, x0, y0, run_info


class PoliTask(BaseTask):
    def __init__(self, tokenizer, candidate_pool, obj_dim, transform=lambda x: x,
                 num_start_examples=None, data_path=None, poli_workers=4, poli_parallelize=True, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim, transform, **kwargs)
        self.op_types = ["sub"]
        self.alphabet: list = None
        self.observer: object = PoliBaseMlFlowObserver(TRACKING_URI)
        self.data_path: str = data_path
        self.assets_pdb_path = None
        self.num_start_examples: int = num_start_examples
        self.poli_parallel: bool = poli_parallelize
        self.poli_workers: int = poli_workers

    def task_setup(self, config, project_root=None, *args, **kwargs):
        global problem_information, f, x0, y0, run_info
        # NOTE: when registering a poli task in hydra, we'll expect that the value to this key can be passed to poli
        self.problem_name = config["task"][POLI_TASK_HYDRA_KEY]
        seed = config.trial_id
        batch_size = config.task.batch_size
        self.num_start_examples = config.task.num_start_examples
        self.data_path = config.task.data_path
        self.assets_pdb_path = None
        caller_info = {
            ALGORITHM: "LAMBO",
            STARTING_N: self.num_start_examples,
            BATCH_SIZE: batch_size,
            }
        if self.problem_name == "foldx_stability_and_sasa": # if additional data i.e. PDBs are needed
            self.assets_pdb_path = list(Path(self.data_path).glob("*/wt_input_Repair.pdb"))
            if self.num_start_examples < 5: # cold-start problem: optimize w.r.t. 1 protein specifically
                self.assets_pdb_paths = [Path(self.data_path) / "1zgo_A" / "wt_input_Repair.pdb"] # pick DsRed specifically.
                # NOTE: lambo requires multiple observations
                for _p_path in list(Path(self.data_path).glob("*/wt_input_Repair.pdb"))[:self.num_start_examples-1]: 
                    if "1zgo_A" in str(_p_path):
                        continue
                    self.assets_pdb_paths.append(_p_path)
            if self.num_start_examples > len(self.assets_pdb_path): # if less data than allowed observations:
                self.num_start_examples = len(self.assets_pdb_path)
                caller_info[STARTING_N] = len(self.assets_pdb_path)
            problem_information, f, x0, y0, run_info = objective_factory.create(
            name=self.problem_name,
            seed=seed,
            caller_info=caller_info,
            wildtype_pdb_path=self.assets_pdb_path,
            observer=self.observer,
            force_register=True,
            parallelize=self.poli_parallel, # NOTE: factory foldx parallelization
            num_workers=self.poli_workers,
            batch_size=batch_size,
            n_starting_points=self.num_start_examples,
            )
        else: # Base case
            problem_information, f, x0, y0, run_info = objective_factory.create(
                name=self.problem_name,
                seed=seed,
                caller_info=caller_info,
                observer=self.observer,
                force_register=True,
                parallelize=self.poli_parallel, # NOTE: factory foldx parallelization
                num_workers=self.poli_workers,
                batch_size=batch_size
            )
        # number of allowed starting sequences:
        if self.num_start_examples is not None:
            x0 = x0[:self.num_start_examples]
            y0 = y0[:self.num_start_examples]
        assert caller_info.get(STARTING_N) == y0.shape[0] == x0.shape[0], "Setting the number of starting examples is illegal for poli tasks!"
        self.alphabet = problem_information.get_alphabet()
        x0_ = x0.copy()
        L = problem_information.get_max_sequence_length()
        # NOTE: rfp from poli is not aligned and has L=np.inf , this we cannot setup via config tasks?
        # assert L == config.task["max_len"], "Inconsistent sequence length task description!"
        if problem_information.sequences_are_aligned():
            assert(np.all([len(x0_[i]) == L for i in range(len(x0_))]))
        # Lambo string candidates does not work with np.arrays -> convert to strings:
        pool_candidates = np.array([StringCandidate("".join(x0_[i]), [], self.tokenizer) for i in range(x0_.shape[0])])
        # NOTE: in this setup: |x0| == |pool_candidates| :
        return pool_candidates, y0, np.array(["".join(x0_[i]) for i in range(x0_.shape[0])]), y0

    def make_new_candidates(self, base_candidates, new_seqs):
        return np.array([StringCandidate(new_seqs[i], [], self.tokenizer) for i in range(len(new_seqs))])

    def _evaluate(self, x, out, *args, **kwargs):
        assert x.ndim == 2
        x_cands, x_seqs, f_vals = [], [], []
        for query_pt in x:
            cand_idx, mut_pos, mut_res_idx, _ = query_pt
            base_candidate = self.candidate_pool[cand_idx]
            mut_res = self.tokenizer.sampling_vocab[mut_res_idx]
            mut_list = [base_candidate.new_mutation(mut_pos, mut_res, mutation_type='sub')]
            candidate = base_candidate.new_candidate(mut_list)
            x_cands.append(candidate)
            x_seqs.append(candidate.mutant_residue_seq)
        x_seqs = np.array(x_seqs).reshape(-1)
        x_cands = np.array(x_cands).reshape(-1)

        out["X_cand"] = x_cands
        out["X_seq"] = x_seqs
        norm_scores = self.transform(self.score(x_cands))
        out["F"] = norm_scores

    def score(self, candidates):
        y = np.zeros([len(candidates), y0.shape[1]])
        for i in range(len(candidates)):
            seq = candidates[i].mutant_residue_seq.upper()
            y[i, ...] = f(np.atleast_2d(list(seq)))
            self.observer.observe(
                np.array([candidates[i].mutant_residue_seq]), y[i:i+1, ...]
            )
        return y
    
    def is_feasible(self, candidates):   
        if self.max_len is None:
            len_feasible = np.ones(candidates.shape).astype(bool)
        else:
            len_feasible = np.array([len(cand) <= self.max_len for cand in candidates]).reshape(-1)
        is_feasible = len_feasible # Default: no PDB
        if self.assets_pdb_path is not None: 
            # if we're working with PDB and FoldX we have to check feasibility
            pdb_feasible = []
            for candidate in candidates:
                try:
                    _ = find_closest_wildtype_pdb_file_to_mutant(self.assets_pdb_path, candidate.mutant_residue_seq.lower())
                except ValueError:
                    is_feasible.append(0)
                    continue
                is_feasible.append(1)
            pdb_feasible = np.array(is_feasible).astype(bool)
            is_feasible = len_feasible * pdb_feasible
        assert is_feasible.shape[0] == candidates.shape[0]
        return is_feasible
    
    def __del__(self):
        """
        Gracefully terminate observer object.
        """
        if self.observer is not None:
            self.observer.finish()