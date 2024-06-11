#!/usr/bin/env bash

set -euxo pipefail

main () {
    cd /code_execution

    if curl --silent --connect-timeout 10 --max-time 12 www.example.com ; then
        echo "Unexpected network connection found"
        return 1
    fi

    data_directory=/code_execution/data
    format_filename=${data_directory}/submission_format.csv
    for ((i=0; i<=5; i++))
    do
      t=$((i * 5))
      if [ -f ${format_filename} ]; then
          echo "found ${format_filename} after $t seconds; data is mounted"
          break
      else
          echo "file ${format_filename} not found after $t seconds, sleeping for 5s ($((i+1)) of 6) to await..."
          sleep 5
      fi
    done
    if [ ! -f ${format_filename} ]; then
      echo "never found ${format_filename}, data is not mounted; exiting"
      return 1
    fi

    expected_filename=main.sh
    submission_files=$(zip -sf ./submission/submission.zip)
    if ! grep -q ${expected_filename}<<<$submission_files; then
        echo "Submission zip archive must include $expected_filename"
    return 1
    fi

    echo Unpacking submission
    unzip ./submission/submission.zip -d ./workdir

    echo Printing submission contents
    find workdir

    pushd workdir
    sh main.sh
    popd

    # test the submission
    pytest --no-header -vv tests/test_submission.py
}

main |& tee "/code_execution/submission/log.txt"
exit_code=${PIPESTATUS[0]}

cp /code_execution/submission/log.txt /tmp/log

exit $exit_code