# List of parameters
#$param_list = "AM AR AROC CMRE EGY EPR EPD ET FLNG FTCO HDSN GSL GOOG JXN KNTK IBM INSW MP MPW MPLX TSLA TRTN OXY V VET ZIM"
$param_list = "ZIM"


# Splitting the string
$param_array = $param_list -split " "

# Loop through the list
foreach ($param in $param_array) {
    # Call the Python program with the current parameter
    & C:/Users/derek/AppData/Local/Microsoft/WindowsApps/python3.9.exe due_diligence.py $param
}

git commit -am 'daily update' 
git push origin main