from utils.tools import run_webcam
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument("--face_reco", "-r", required=False, action="store_true",
                    help="Active la fonctionnalité de reconnaissance faciale")
    ap.add_argument("--face_detect", "-d", required=False, action="store_true",
                    help="Active la fonctionnalité de détection faciale")
    args = vars(ap.parse_args())

    if args["face_reco"] and args["face_detect"]:
        raise Exception("[Argument invalide] Entrer soit '-r', soit '-d', mais pas les deux en même temps.")
    elif not args["face_reco"] and not args["face_detect"]:
        raise Exception("[Argument invalide] Entrer soit '-r', soit '-d'.")
    else:
        run_webcam(args["face_reco"], args["face_detect"])
