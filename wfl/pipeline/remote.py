import sys
import os

from wfl.configset import ConfigSet_in, ConfigSet_out
from .utils import grouper, RemoteInfo
from .pool import do_in_pool

from expyre import ExPyRe


def do_remotely(remote_info, hash_ignore=[], chunksize=1, iterable=None, configset_out=None, op=None, iterable_arg=0,
                skip_failed=True, initializer=None, initargs=None, args=[], kwargs={}, quiet=False):
    """run tasks as series of remote jobs

    Parameters
    ----------
    remote_info: RemoteInfo or dict
        object with all information on remote job, including system, resources, job chunksize, etc, or dict of kwargs for its constructor
    quiet: bool, default False
        do not output (to stderr) progress info

    See pipeline.iterable_loop() for other args
    """
    if ExPyRe is None:
        raise RuntimeError('Cannot run as remote jobs since expyre module could not be imported')

    if not isinstance(remote_info, RemoteInfo):
        remote_info = RemoteInfo(**remote_info)

    if remote_info.job_chunksize < 0:
        remote_info.job_chunksize = -remote_info.job_chunksize * chunksize

    if isinstance(iterable, ConfigSet_in):
        items_inputs_generator = grouper(remote_info.job_chunksize, ((item, iterable.get_current_input_file()) for item in iterable))
    else:
        items_inputs_generator = grouper(remote_info.job_chunksize, ((item, None) for item in iterable))

    # create all jobs (count on expyre detection of identical jobs to avoid rerunning things unnecessarily)
    xprs = []
    input_files = []
    for chunk_i, items_gen in enumerate(items_inputs_generator):
        items = []
        for (item, cur_input_file) in items_gen:
            items.append(item)
            input_files.append(cur_input_file)

        job_name = remote_info.job_name + f'_chunk_{chunk_i}'
        if not quiet:
            sys.stderr.write(f'Creating job {job_name}\n')

        if isinstance(iterable, ConfigSet_in):
            job_iterable = ConfigSet_in(input_configs=items)
        else:
            job_iterable = items
        co = ConfigSet_out()
        # remote job will have to set npool appropriately for its node
        # ignore configset out for hashing of inputs, since that doesn't affect function
        # calls that have to happen (also it's not repeatable for some reason)
        xprs.append(ExPyRe(name=job_name, pre_run_commands=remote_info.pre_cmds, post_run_commands=remote_info.post_cmds,
                            hash_ignore=hash_ignore + ['configset_out'],
                            env_vars=remote_info.env_vars, input_files=remote_info.input_files,
                            output_files=remote_info.output_files, function=do_in_pool,
                            kwargs={'npool': None, 'chunksize': chunksize, 'iterable': job_iterable,
                                    'configset_out': co, 'op': op, 'iterable_arg': iterable_arg,
                                    'skip_failed': skip_failed, 'initializer': initializer,
                                    'initargs': initargs, 'args': args, 'kwargs': kwargs}))

    # start jobs (shouldn't do anything if they've already been started)
    for xpr in xprs:
        if not quiet:
            sys.stderr.write(f'Starting job for {xpr.id}\n')
        xpr.start(resources=remote_info.resources, system_name=remote_info.sys_name,
                  exact_fit=remote_info.exact_fit, partial_node=remote_info.partial_node)

    # gather results and write them to original configset_out
    configset_out.pre_write()
    at_i = 0
    for xpr in xprs:
        if not quiet:
            sys.stderr.write(f'Gathering results for {xpr.id}\n')
        ats_out, stdout, stderr = xpr.get_results(timeout=remote_info.timeout, check_interval=remote_info.check_interval)
        for at in ats_out.group_iter():
            configset_out.write(at, from_input_file=input_files[at_i])
            at_i += 1
        sys.stdout.write(stdout)
        sys.stderr.write(stderr)
    configset_out.end_write()

    if 'WFL_AUTOPARA_REMOTE_NO_MARK_PROCESSED' not in os.environ:
        # mark as processed only after configset_out has been finished
        for xpr in xprs:
            xpr.mark_processed()

    return configset_out.to_ConfigSet_in()