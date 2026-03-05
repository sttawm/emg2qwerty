# Team Lead Guide

## Initial Setup (One-Time)

```bash
cd bootstrap

# 1. Create shared project and buckets
./setup_shared_resources.sh

# 2. Upload dataset
./upload_data.sh /path/to/data

# 3. Grant yourself access
./grant_access.sh your-email@gmail.com

# 4. Setup your personal project
./setup_teammate.sh
```

**Shared resources created:**
- Data: `gs://emg2qwerty-team-data`
- Logs: `gs://emg2qwerty-team-logs`
- Config: `bootstrap/shared_config.env` (commit to git)

---

## Adding Teammates

```bash
cd bootstrap
./grant_access.sh teammate@gmail.com
```

**Then send them:**
```
Repo: https://github.com/sttawm/emg2qwerty
Setup: See TEAMMATE_SETUP.md
```

**They run:** `./bootstrap/setup_teammate.sh` and start training.

---

## Remove Access

```bash
gsutil iam ch -d user:teammate@gmail.com:objectViewer gs://emg2qwerty-team-data
gsutil iam ch -d user:teammate@gmail.com:objectAdmin gs://emg2qwerty-team-logs
```
