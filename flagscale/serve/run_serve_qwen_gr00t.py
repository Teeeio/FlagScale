from flagscale.serve.run_serve_vla import main, parse_config

if __name__ == "__main__":
    parsed_cfg = parse_config()
    main(parsed_cfg["serve"][0])
