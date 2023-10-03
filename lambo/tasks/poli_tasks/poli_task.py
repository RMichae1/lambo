from multiprocessing import pool
import numpy as np
from pathlib import Path

from poli import objective_factory
from poli.core.util.external_observer import ExternalObserver

from lambo.candidate import StringCandidate
from lambo.tasks.base_task import BaseTask
from lambo.tasks.poli_tasks import ALGORITHM, STARTING_N, BATCH_SIZE
from lambo.tasks.poli_tasks import POLI_TASK_HYDRA_KEY

from corel.observers.poli_base_logger import PoliBaseMlFlowObserver


global problem_information, f, x0, y0, run_info


class PoliTask(BaseTask):
    def __init__(self, tokenizer, candidate_pool, obj_dim, transform=lambda x: x,
                 num_start_examples=None, data_path=None, poli_workers=4, poli_parallelize=True, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim, transform, **kwargs)
        self.op_types = ["sub"]
        self.alphabet: list = None
        self.observer: object = PoliBaseMlFlowObserver("file:/Users/rcml/corel/results/mlruns/")
        self.data_path: str = data_path
        self.num_start_examples: int = num_start_examples
        self.poli_parallel: bool = poli_parallelize
        self.poli_workers: int = poli_workers

    def task_setup(self, config, project_root=None, *args, **kwargs):
        global problem_information, f, x0, y0, run_info
        # NOTE: when registering a poli task in hydra, we'll expect that the value to this key can be passed to poli
        problem_name = config["task"][POLI_TASK_HYDRA_KEY]
        seed = config.trial_id
        batch_size = config.task.batch_size
        self.num_start_examples = config.task.num_start_examples
        self.data_path = config.task.data_path
        caller_info = {
            ALGORITHM: "LAMBO",
            STARTING_N: self.num_start_examples,
            BATCH_SIZE: batch_size,
            }
        if problem_name == "foldx_stability_and_sasa": # if additional data i.e. PDBs are needed
            assets_pdb_path = list(Path(self.data_path).glob("*/wt_input_Repair.pdb"))
            if self.num_start_examples > len(assets_pdb_path): # if less data than allowed observations:
                self.num_start_examples = len(assets_pdb_path)
                caller_info[STARTING_N] = len(assets_pdb_path)
            problem_information, f, x0, y0, run_info = objective_factory.create(
            name=problem_name,
            seed=seed,
            caller_info=caller_info,
            wildtype_pdb_path=assets_pdb_path,
            observer=self.observer,
            force_register=True,
            parallelize=self.poli_parallel, # NOTE: factory foldx parallelization
            num_workers=self.poli_workers,
            batch_size=batch_size
            )
        else: # Base case
            problem_information, f, x0, y0, run_info = objective_factory.create(
                name=problem_name,
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
            seq = candidates[i].wild_residue_seq.lower()
            y[i, ...] = f(np.array([seq]))
            self.observer.observe(
                np.array([candidates[i].mutant_residue_seq]), y[i:i+1, ...]
            )
            # lambo appends bof and eof tokens
            # y[i] = f(np.array([self.alphabet[seq[n]] for n in range(1, len(seq)-1)]).reshape(1, -1))
        return y
    
    def __del__(self):
        """
        Gracefully terminate observer object.
        """
        if self.observer is not None:
            self.observer.finish()