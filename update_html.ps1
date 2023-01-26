# List of parameters
#$param_list = "AR AROC CMRE EGY EPR EPD ET FLNG FTCO HDSN GSL KNTK IBM INSW MP MPW MPLX TRTN OXY V VET ZIM"
$param_list = "IBM"


# Splitting the string
$param_array = $param_list -split " "

# Loop through the list
foreach ($param in $param_array) {
    # Call the Python program with the current parameter
    & C:/Users/derek/AppData/Local/Microsoft/WindowsApps/python3.9.exe due_diligence.py $param
}

git commit -am 'daily update' 
git push origin main