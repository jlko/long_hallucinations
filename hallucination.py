"""Main script to run hallucination experiments."""
import os
import argparse
import logging

import pickle
from collections import defaultdict

import wandb

import utils
from utils import log_w_indent
import models
from data import data

# pylint: disable=invalid-name

PROJECT = 'long_hallu'
FILENAME = 'results.pkl'


def main(args):

    # <Inits>
    user = os.getenv('USER')
    slurm_jobid = os.getenv('SLURM_JOB_ID')
    wandb_path = f'/scratch-ssd/{user}/long_hallu'
    os.system(f'mkdir -p {wandb_path}')
    wandb.init(
        project=PROJECT if not args.debug else f'{PROJECT}_debug',
        dir=wandb_path,
        config=args,
        notes=slurm_jobid,
    )
    logging.info('New run with config:')
    logging.info(args)
    logging.info('SLURM_JOB_ID: %s', slurm_jobid)
    logging.info('Wandb setup finished.')

    os.environ['HALLU_RESTORE_ID'] = str(args.restore_from_wandb_id)
    os.environ['HALLU_RESTORE_STAGES'] = '-'.join(args.restore_stages)

    if (run := args.restore_from_wandb_id) is not None:
        logging.info('Restoring results from run %s', run)
        restored, restored_config = utils.wandb_restore(run, FILENAME)
        restored = restored['export_predictions']
    else:
        restored = defaultdict(list)
        restored_config = dict()

    kwargs = dict(
        n_questions=args.n_questions, n_regenerate=args.n_regenerate,
        n_stochastic_questions=args.n_stochastic_questions,
        restored=restored, restore_stages=args.restore_stages,
        accept_restore_failure=args.accept_restore_failure,
        entailment_type=args.entailment_type)
    model = models.all_models[args.model](**kwargs)

    log_prompts = model.get_all_prompts_for_log()

    if args.wait:
        wait = lambda: input('wait')  # pylint: disable = unnecessary-lambda-assignment
    else:
        wait = lambda: 0  # pylint: disable = unnecessary-lambda-assignment

    logging.info('Prompts are:')
    for key, value in log_prompts.items():
        logging.info('%s: %s', key, value)

    results = dict(
        prompts=log_prompts, restored_config=restored_config,
        questions=dict(), metrics=dict())

    all_labels, all_uncertainties = [], []
    # Iterate over different individuals in FactualBio.
    for datum in data:
        didx, user_question, init_reply, init_reply_labels, facts, facts_labels = datum

        propositions, labels = facts, facts_labels

        results['questions'][f'datum-{didx}'] = dict(
            user_question=user_question, init_reply=init_reply,
            init_reply_labels=init_reply_labels, facts=facts,
            facts_labels=facts_labels, uncertainties=[],
            propositions=dict())
        ru = results['questions'][f'datum-{didx}']
        log_w_indent(f'User question {didx}: `{user_question}`', indent=0)

        # Iterate over extracted `facts` for each individual.
        for pidx, (proposition, label) in enumerate(zip(propositions, labels)):

            text_so_far = ' '.join(propositions[:pidx]) if pidx > 0 else None

            log_w_indent(
                f'Currently dealing with proposition {pidx}: {proposition}', 1)

            results['questions'][f'datum-{didx}']['propositions'][f'prop-{pidx}'] = {}
            # Estimate uncertainty of proposition.
            uncertainty = model.check_truth(
                rp=ru['propositions'][f'prop-{pidx}'],
                wait=wait,
                data=dict(
                    didx=didx,
                    user_question=user_question,
                    proposition=proposition,
                    text_so_far=text_so_far)
            )

            log_w_indent(f'Final uncertainty for proposition {proposition}: {uncertainty}', 1)
            ru['uncertainties'].append(uncertainty)
            all_uncertainties.append(uncertainty)
            all_labels.append(label)

            wait()

        # Compute metrics.
        early = didx + 1 == args.num_data
        finish = early or (didx + 1 == len(data))
        if args.intermediate_export or finish:
            results['metrics'] = utils.get_metrics(all_labels, all_uncertainties)
            log_w_indent(f'Results: {results["metrics"]}', 0)
            wandb.log(results['metrics'])

            log_w_indent('Final generation with uncertainty', 0)
            for pidx, proposition in enumerate(propositions):
                uncertainties = ru['uncertainties'][pidx]
                log_w_indent(f'{uncertainties:.3f} {proposition}', 0)

            out = dict(results=results, export_predictions=model.export_predictions)
            with open(f'{wandb.run.dir}/{FILENAME}', 'wb') as file:
                pickle.dump(out, file)
            wandb.save(FILENAME)

        if early:
            logging.warning('Ending eval early!')
            break

    log_w_indent('Run finished', 0)


if __name__ == '__main__':
    utils.setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False,
        help="Keep default wandb clean.")
    parser.add_argument(
        "--wait", action=argparse.BooleanOptionalAction, default=False,
        help="Step through execution with pauses.")
    parser.add_argument(
        "--intermediate_export", action=argparse.BooleanOptionalAction, default=True,
        help="Step through execution with pauses.")
    parser.add_argument(
        "--model", type=str, default='QADebertaEntailment',
        help="Set of prompts to use.")
    parser.add_argument(
        "--n_questions", type=str, default='three',
        help="Number of questions to ask per proposition.")
    parser.add_argument(
        "--n_stochastic_questions", type=int, default=2,
        help="Number of times we generate questions.")
    parser.add_argument(
        "--n_regenerate", type=int, default=3,
        help="Number of answers per question.")
    parser.add_argument(
        "--num_data", type=int, default=int(1e19),
        help="Number of datapoints to analyse.")
    parser.add_argument(
        "--entailment_type", type=str, default='lax',  # or strict
        help="Lax or strict entailment.")
    parser.add_argument(
        "--restore_from_wandb_id", type=str, default=None,
        help=(
                "Restore (or copy) parts of a previous run. Need to also set "
                "`--restore_stages` appropriately."))
    parser.add_argument(
        "--restore_stages", default=[], nargs='*', type=str,
        help=(
            "Which stages to restore. Choices = "
            "[{model.gen_qs}, {model.answer_qs}, {model.equivalence}]"))
    parser.add_argument(
        "--accept_restore_failure", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Safely recover from restore failures. Use with care, as usually "
            "restore failures indicate you might not be restoring what you "
            "think you are. An exception to this is adding more questions to a run!"))

    main(parser.parse_args())
