# Description: This script checks if specific network drives are mounted on the local machine.
param (
    [string]$Username,
    [string]$Password
)

# --- CONFIGURATION ---
$drivesToMount = @{
    "F:" = "\\cup\falkner";
    "M:" = "\\cup\murthy"
}
$statusFile = "C:\temp\remount_status.txt"

# clear old status file
if (Test-Path $statusFile) {
    Remove-Item $statusFile
}

# get currently mounted drive letters
$mountedDrives = (Get-PSDrive -PSProvider "FileSystem" | Select-Object -ExpandProperty Name).ToUpper()

# check each drive and mount if necessary
$drivesToMount.GetEnumerator() | ForEach-Object {
    $driveLetterWithColon = $_.Name
    $driveLetterOnly = $driveLetterWithColon.Replace(":", "").ToUpper()
    $path = $_.Value

    if ($driveLetterOnly -in $mountedDrives) {
        "[**Local mount check**]'$driveLetterWithColon' is already mounted on this PC." | Add-Content -Path $statusFile
    } else {
        try {
            # use the provided credentials to mount the drive
            $netUseCommand = "net use $($driveLetterWithColon.ToLower()) `"$path`" /user:$Username $Password /persistent:yes"
            Invoke-Expression $netUseCommand | Out-Null
            "[**Local mount check**]'$driveLetterWithColon' has now been mounted on this PC." | Add-Content -Path $statusFile
        } catch {
            "[**Local mount check**]FAILED to mount '$driveLetterWithColon'. Error: $_" | Add-Content -Path $statusFile
        }
    }
}