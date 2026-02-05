#!/usr/bin/env python3
"""Convert notebook performancedb_dora.ipynb into a runnable script.

Usage:
    python performancedb_dora.py --dora postmdt-1926 --db ./performance.db
    python performancedb_dora.py --dora postmdt-1926 --dora otherID --db /path/to/performance.db

The script queries doralite for metadata, parses ascii output archives to collect
performance metrics, and writes a few summary tables into the sqlite database.
"""
from typing import Iterable, Any
import argparse
import re
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import glob
import subprocess
import os
import fnmatch
import sqlite3
import logging

# Optional import used by the original notebook
try:
    import doralite
except Exception:
    doralite = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def getClocksFMSout(DIR, TAG='', loud=True):
    """Get the clocks that are printed in the fms.out inside ascii tar files.

    Returns:
        modelYear, RunTimemaxClock, MainmaxClock, OCNmaxClock, ATMmaxClock,
        NumberBergsEnd, MOM_MaxCFL_YR, MOM_MaxTrunc_YR, SIS_MaxTrunc_YR,
        RunDate, HiWaterMark
    """
    FILES = sorted(glob.glob(os.path.join(DIR, 'ascii', f'{TAG}*.ascii_out.tar')))

    OCNmaxClock = []
    ATMmaxClock = []
    MainmaxClock = []
    RunTimemaxClock = []
    NumberBergsEnd = []
    MOM_MaxCFL_YR = []
    MOM_MaxTrunc_YR = []
    SIS_MaxTrunc_YR = []
    modelYear = []
    RunDate = []
    HiWaterMark = []

    for file in FILES:
        try:
            tar = tarfile.open(file)
        except (FileNotFoundError, tarfile.ReadError) as e:
            logging.warning("Could not open tar %s: %s", file, e)
            continue

        TAGlocal = os.path.basename(file)[:8]
        if loud:
            logging.info(TAGlocal)
        try:
            FH = tar.extractfile('./' + TAGlocal + '.fms.out')
            if FH is None:
                logging.warning('fms.out not found inside %s', file)
                continue
            lines = FH.readlines()
            FH.close()
        finally:
            tar.close()

        modelYear.append(TAGlocal[:4])
        MOM_date = []
        MOM_MaxCFL = []
        MOM_Trunc = [0]
        SIS_Trunc = [0]

        for raw in lines:
            line = raw.decode()

            m = re.search(r"Entering coupler_init at (\d*).*", line)
            if m:
                RunDate.append(m.group(1))

            m = re.search(r"^KID.* write_restart berg .* #= \s*(\d*).*", line)
            if m:
                NumberBergsEnd.append(m.group(1))

            m = re.search(r"HiWaterMark", line)
            if m:
                try:
                    HiWaterMark.append(np.double(line[65:75]))
                except Exception:
                    pass

            mTotRun = re.search(r"Total runtime", line)
            if mTotRun:
                try:
                    RunTimemaxClock.append(np.double(line[71:85]))
                except Exception:
                    pass

            mMAIN = re.search(r"Main loop ", line)
            if mMAIN:
                try:
                    MainmaxClock.append(np.double(line[57:71]))
                except Exception:
                    pass

            mOCN = re.search(r"OCN           ", line)
            if mOCN:
                try:
                    OCNmaxClock.append(np.double(line[57:71]))
                except Exception:
                    pass

            mATM = re.search(r"ATM           ", line)
            if mATM:
                try:
                    ATMmaxClock.append(np.double(line[57:71]))
                except Exception:
                    pass

            m = re.search(r"MaxCFL", line)
            if m:
                MOM_date.append((line[11:30]))
                try:
                    MOM_MaxCFL.append(np.double(line[63:70]))
                except Exception:
                    pass

            m = re.search(r"Truncations", line)
            n = re.search(r"Sea Ice Truncations", line)
            if (m and n):
                try:
                    SIS_Trunc.append(np.double(line[39:]))
                except Exception:
                    pass
            elif (m):
                try:
                    MOM_Trunc.append(np.double(line[67:]))
                except Exception:
                    pass

        if MOM_MaxCFL:
            MOM_MaxCFL_YR.append(max(MOM_MaxCFL))
        else:
            MOM_MaxCFL_YR.append(0)
        MOM_MaxTrunc_YR.append(max(MOM_Trunc) if MOM_Trunc else 0)
        SIS_MaxTrunc_YR.append(max(SIS_Trunc) if SIS_Trunc else 0)

    return (
        modelYear,
        RunTimemaxClock,
        MainmaxClock,
        OCNmaxClock,
        ATMmaxClock,
        NumberBergsEnd,
        MOM_MaxCFL_YR,
        MOM_MaxTrunc_YR,
        SIS_MaxTrunc_YR,
        RunDate,
        HiWaterMark,
    )


def getPropsFromStdout(DIR, TAG='', pattern='./cm5*.o*', itemList=None):
    """Get some of the properties that are printed in the stdout and are missing from fms.out"""
    if itemList is None:
        itemList = []

    FILES = sorted(glob.glob(os.path.join(DIR, 'ascii', f'{TAG}*.ascii_out.tar')))
    prop = {}

    for file in FILES:
        logging.info('Processing file: %s', file)
        try:
            with tarfile.open(file, 'r') as tar:
                for member in tar.getmembers():
                    if fnmatch.fnmatch(member.name, pattern):
                        FH = tar.extractfile(member.name)
                        if FH is None:
                            continue
                        lines = FH.readlines()
                        FH.close()
        except FileNotFoundError:
            logging.warning('Tar archive not found at %s', file)
            continue
        except tarfile.ReadError:
            logging.warning('Could not read tar archive at %s', file)
            continue

        for raw in lines:
            line = raw.decode()
            for key in ['atm_ranks', 'atm_threads', 'ocn_ranks', 'ocn_threads'] + itemList:
                regex = r'set -r ' + re.escape(key) + r' = (\S*)'
                m = re.match(regex, line, re.IGNORECASE)
                if m:
                    prop[key] = m.group(1)
                    break

    if all(k in prop for k in ('atm_ranks', 'atm_threads', 'ocn_ranks', 'ocn_threads')):
        try:
            prop['TotalCores'] = int(prop['atm_ranks']) * int(prop['atm_threads']) + int(prop['ocn_ranks']) * int(prop['ocn_threads'])
        except Exception:
            prop['TotalCores'] = None
    else:
        prop['TotalCores'] = None

    return prop


def historySizeGB(DIR, loud=True):
    history_path = os.path.join(DIR, 'history', '00010101.nc.tar')
    try:
        history_size = os.path.getsize(history_path)
        return history_size / 1024.0 / 1024.0 / 1024.0
    except Exception:
        return None


def dirSizeGB(DIR):
    total_size = 0
    for path, dirs, files in os.walk(DIR):
        for f in files:
            fp = os.path.join(path, f)
            try:
                total_size += os.path.getsize(fp)
            except Exception:
                pass
    return total_size / 1024.0 / 1024.0 / 1024.0


def numberOfYears(DIR):
    try:
        process = subprocess.run("ls -1 " + os.path.join(DIR, 'history') + " | wc -l", capture_output=True, text=True, shell=True)
        return int(process.stdout)
    except Exception:
        return None


def getPlatform(DIR):
    match = re.search(r"(ncrc\d+-[a-zA-Z0-9]+)-" ,DIR)
    return match.group(1)

def insert_two_columns(
    db_path: str,
    table_name: str,
    col1_name: str,
    col2_name: str,
    list1: Iterable[Any],
    list2: Iterable[Any],
    if_exists: str = "replace",
) -> None:
    """Insert two parallel lists into a sqlite table.

    Args:
        db_path: path to sqlite database file (will be created if missing).
        table_name: name of the table to create/insert into.
        col1_name: column name for values from `list1`.
        col2_name: column name for values from `list2`.
        list1: iterable of values for first column.
        list2: iterable of values for second column.
        if_exists: "replace" to drop & recreate table, "append" to keep existing rows.

    Raises:
        ValueError: if the two input lists have different lengths.
    """
    l1 = list(list1)
    l2 = list(list2)
    if len(l1) != len(l2):
        raise ValueError("list1 and list2 must have the same length")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if if_exists == "replace":
            cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')

        cur.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ("{col1_name}", "{col2_name}")')
        cur.executemany(
            f'INSERT INTO "{table_name}" ("{col1_name}", "{col2_name}") VALUES (?, ?)',
            list(zip(l1, l2)),
        )
        conn.commit()
    finally:
        conn.close()


def process_dora(doraID: str, db_path: str, replace_tables: bool = True):
    if doralite is None:
        logging.error('doralite module not available. Cannot query dora metadata.')
        return

    logging.info('Querying dora metadata for %s', doraID)
    experiment = doralite.dora_metadata(doraID)
    pathHist = experiment.get('pathHistory')
    if not pathHist:
        logging.error('pathHistory not found for %s', doraID)
        return

    expName = experiment.get('expName', '')
    archDir = os.path.join(pathHist, '..')
    # Normalize path
    archDir = os.path.normpath(archDir) + os.sep

    logging.info('Processing fms.out files in archDir %s for experiment %s', archDir, expName)
    (
        YR,
        Totalruntime,
        main,
        ocn,
        atm,
        NumBergs,
        MOM_MaxCFL,
        MOM_MaxTrunc,
        SIS_MaxTrunc,
        RunDate,
        HiWaterMark,
    ) = getClocksFMSout(archDir, loud=False)


    experiment['doraID'] = doraID
    experiment['platform'] = getPlatform(archDir)
    experiment['archDir'] = archDir
    experiment['TotalStorageSizeGB'] = dirSizeGB(archDir)
    experiment['AnnualHistorySizeGB'] = historySizeGB(archDir)
    experiment['NumberOfYears'] = numberOfYears(archDir)
    try:
        experiment['HoursPerYear'] = float(np.average(np.array(Totalruntime[1])) / 3600)
    except Exception:
        experiment['HoursPerYear'] = None
    experiment['model_year'] = np.double(YR) if YR else []
    experiment['HiWaterMark'] = np.double(HiWaterMark) if HiWaterMark else []
    experiment['Totalruntime'] = np.double(Totalruntime) if Totalruntime else []
    experiment['Main_clock_max'] = np.double(main) if main else []
    experiment['OCN_clock_max'] = np.double(ocn) if ocn else []
    experiment['ATM_clock_max'] = np.double(atm) if atm else []
    experiment['NumberOfIceBergs'] = np.double(NumBergs) if NumBergs else []
    experiment['MOM_MaxCFL'] = np.double(MOM_MaxCFL) if MOM_MaxCFL else []
    experiment['MOM_MaxTrunc'] = np.double(MOM_MaxTrunc) if MOM_MaxTrunc else []
    experiment['SIS_MaxTrunc'] = np.double(SIS_MaxTrunc) if SIS_MaxTrunc else []
    experiment['RunDate'] = RunDate
    experiment['RunDateSpan'] = (RunDate[0] + ' - ' + RunDate[-1]) if RunDate else ''

    # Query and Add properties that are in stdout and not in fms.out
    #props = getPropsFromStdout(archDir, TAG='00010101', pattern='./' + expName + '*.o*')
    #for key in ['atm_ranks', 'atm_threads', 'ocn_ranks', 'ocn_threads', 'TotalCores']:
    #    experiment[key] = props.get(key)

    #Print summary
    import pandas as pd
    df = pd.DataFrame(experiment)
    print(df.loc[0, ['expName','NumberOfYears','TotalStorageSizeGB','AnnualHistorySizeGB','HoursPerYear','platform','RunDateSpan']])
    
    logging.info('Writing results to sqlite database at %s', db_path)
    # Insert several tables into sqlite db
    if replace_tables:
        mode = 'replace'
    else:
        mode = 'append'

    # Prepare numeric lists (ensure plain python lists)
    years = list(map(int, map(float, experiment.get('model_year', [])))) if len(experiment.get('model_year', [])) > 0 else []

    def safe_list(values):
        try:
            return list(map(float, values))
        except Exception:
            return list(values)

    insert_two_columns(db_path, 'MaxTruncations_MOM6', 'year', 'value', years, safe_list(experiment.get('MOM_MaxTrunc', [])), if_exists=mode)
    insert_two_columns(db_path, 'MaxTruncations_SIS2', 'year', 'value', years, safe_list(experiment.get('SIS_MaxTrunc', [])), if_exists=mode)
    insert_two_columns(db_path, 'clock_max_Main_hours', 'year', 'value', years, [v / 3600.0 for v in safe_list(experiment.get('Main_clock_max', []))], if_exists=mode)
    insert_two_columns(db_path, 'clock_max_OCN_hours', 'year', 'value', years, [v / 3600.0 for v in safe_list(experiment.get('OCN_clock_max', []))], if_exists=mode)
    insert_two_columns(db_path, 'clock_max_ATM_hours', 'year', 'value', years, [v / 3600.0 for v in safe_list(experiment.get('ATM_clock_max', []))], if_exists=mode)
    insert_two_columns(db_path, 'NumberOfIceBergs', 'year', 'value', years, safe_list(experiment.get('NumberOfIceBergs', [])), if_exists=mode)

    # Also write a simple metadata table
#    conn = sqlite3.connect(db_path)
#    try:
#        cur = conn.cursor()
#        cur.execute('CREATE TABLE IF NOT EXISTS dora_metadata (doraID, key, value)')
#        for key, val in experiment.items():
#            try:
#                cur.execute('INSERT INTO dora_metadata (doraID, key, value) VALUES (?, ?, ?)', (doraID, str(key), str(val)))
#            except Exception:
#                logging.debug('Could not insert metadata key %s', key)
#        conn.commit()
#    finally:
#        conn.close()

    logging.info('Finished processing %s. Results written to %s', doraID, db_path)


def main():
    parser = argparse.ArgumentParser(description='Process dora experiments and write performance tables to sqlite DB')
    parser.add_argument('--dora', required=True, action='append', help='DORA id (can be repeated)')
    parser.add_argument('--db', required=True, help='Path to sqlite database file (will be created if missing)')
    parser.add_argument('--append', action='store_true', help='Append to existing tables instead of replacing')
    args = parser.parse_args()

    for d in args.dora:
        process_dora(d, args.db, replace_tables=not args.append)


if __name__ == '__main__':
    main()
