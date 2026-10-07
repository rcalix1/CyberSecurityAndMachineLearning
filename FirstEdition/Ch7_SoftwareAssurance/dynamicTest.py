import subprocess

program = "/bin/ls"
log_file = "execution_log.txt"

with open(log_file, "w") as log:

    subprocess.run(
        ["strace", "-f", program],
        stderr=log,
        timeout=30
    )

print("Dynamic analysis complete.")
print("Log saved to:", log_file)
