"""Allow-listed accelerator quota data from the official Kaggle API."""
import math
import re


def duration_seconds(value):
    if isinstance(value,bool) or value is None:return None
    if isinstance(value,dict):
        try:value=float(value['seconds'])+float(value.get('nanos',0))/1e9
        except (KeyError,TypeError,ValueError):return None
    elif isinstance(value,str):
        if not re.fullmatch(r'\d+(?:\.\d+)?s?',value):return None
        value=float(value.rstrip('s'))
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:return None
    return float(value)


def normalize_quota(payload):
    result={'available':False,'source':'kaggle_api'}
    if not isinstance(payload,dict):return result
    for name in ('gpu','tpu'):
        raw=payload.get(name+'Quota')
        if not isinstance(raw,dict):continue
        values={key:duration_seconds(raw.get(field)) for key,field in (
            ('used_seconds','timeUsed'),('reserved_seconds','timeReserved'),
            ('total_seconds','totalTimeAllowed'),('minimum_seconds','minimumTimeAllowed'))}
        used,reserved,total=(values[k] for k in ('used_seconds','reserved_seconds','total_seconds'))
        # Remaining is derived from official values, never a hardcoded weekly quota.
        values['remaining_seconds']=max(0,total-used-reserved) if None not in (used,reserved,total) else None
        result[name]=values
        if name=='gpu' and total is not None:result['available']=True
    reset=payload.get('quotaRefreshTime')
    if isinstance(reset,str) and len(reset)<100:result['reset_at']=reset
    return result
