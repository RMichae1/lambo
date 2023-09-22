import numpy as np

from poli import objective_factory
from poli.core.util.external_observer import ExternalObserver

from lambo.candidate import StringCandidate
from lambo.tasks.base_task import BaseTask
from lambo.tasks.poli_tasks import ALGORITHM, STARTING_N, BATCH_SIZE
from lambo.tasks.poli_tasks import POLI_TASK_HYDRA_KEY


global problem_information, f, x0, y0, run_info
global observer


class PoliTask(BaseTask):
    def __init__(self, tokenizer, candidate_pool, obj_dim, transform=lambda x: x,
                 num_start_examples=1024, **kwargs):
        super().__init__(tokenizer, candidate_pool, obj_dim, transform, **kwargs)
        self.op_types = ["sub"]
        self.num_start_examples = num_start_examples
        self.alphabet = None

    def task_setup(self, config, project_root=None, *args, **kwargs):
        global problem_information, f, x0, y0, run_info
        global observer
        # NOTE: when registering a poli task in hydra, we'll expect that the value to this key can be passed to poli
        problem_name = config["task"][POLI_TASK_HYDRA_KEY]
        seed = config.trial_id
        caller_info = {
            ALGORITHM: "LAMBO",
            STARTING_N: self.num_start_examples,
            BATCH_SIZE: config.task.batch_size,
            }
        problem_information, f, x0, y0, run_info = objective_factory.create(problem_name,
                                                                            caller_info=caller_info,
                                                                            seed=seed)
        observer = ExternalObserver()
        observer.initialize_observer(
            problem_information, caller_info, x0, y0, seed,
        )
        if "num_start_examples" in config.task.keys():
            assert config.task["num_start_examples"] == y0.shape[0], "Setting the number of starting examples is illegal for poli tasks!"
        self.alphabet = problem_information.get_alphabet()
        x0_ = x0
        L = problem_information.get_max_sequence_length()
        assert L == config.task["max_len"], "Inconsistent sequence length task description!"
        if problem_information.sequences_are_aligned():
            assert(np.all([len(x0_[i]) == L for i in range(len(x0_))]))
        pool_candidates = np.array([StringCandidate(x0_[i], [], self.tokenizer) for i in range(x0_.shape[0])])
        return pool_candidates, y0, x0_, y0

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
            observer.observe(
                np.array([candidates[i].mutant_residue_seq]), y[i:i+1, ...]
            )
            # lambo appends bof and eof tokens
            # y[i] = f(np.array([self.alphabet[seq[n]] for n in range(1, len(seq)-1)]).reshape(1, -1))
        return y