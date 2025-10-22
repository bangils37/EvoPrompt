from BBH.utils import  set_seed
from BBH.args import parse_args
from BBH.llm_client import llm_init


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    client = None
    llm_config = llm_init(f"./auth.yaml", args.llm_type, args.setting)
    if args.evo_mode == 'de':
        from BBH.evoluter import DEEvoluter
        evoluter = DEEvoluter(args, llm_config, client)
        evoluter.evolute()
    elif args.evo_mode == 'ga':
        from BBH.evoluter import GAEvoluter
        evoluter = GAEvoluter(args, llm_config, client)
        evoluter.evolute()
    elif args.evo_mode == 'ape':
        from BBH.evoluter import ParaEvoluter
        evoluter = ParaEvoluter(args, llm_config, client)
        evoluter.evolute()
