# Snapshot of what else is running on the box (sibling-process noise, spec D9): top processes by
# working set, every python.exe command line, and the free/total physical memory and swap.
# usage: powershell -NoProfile -ExecutionPolicy Bypass -File siblings.ps1
$os = Get-CimInstance Win32_OperatingSystem
"physical: free {0:N1} GiB of {1:N1} GiB; pagefile in use {2:N1} GiB" -f ($os.FreePhysicalMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB), ((Get-CimInstance Win32_PageFileUsage | Measure-Object -Property CurrentUsage -Sum).Sum / 1024)
""
"-- python.exe processes (pid, working set GiB, command line)"
Get-CimInstance Win32_Process -Filter "name='python.exe'" | ForEach-Object {
    "{0} {1:N2} GiB {2}" -f $_.ProcessId, ($_.WorkingSetSize / 1GB), $_.CommandLine
}
""
"-- top 15 processes by working set"
Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 15 | ForEach-Object {
    "{0,-28} pid {1,-7} {2,8:N2} GiB" -f $_.ProcessName, $_.Id, ($_.WorkingSet64 / 1GB)
}
