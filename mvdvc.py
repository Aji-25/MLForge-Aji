import argparse
import sys
from mvdvc_core import MvDvc

def main():
    parser = argparse.ArgumentParser(description="mvdvc: A mini DVC-like tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # init
    parser_init = subparsers.add_parser("init", help="Initialize mvdvc")
    
    # add
    parser_add = subparsers.add_parser("add", help="Add file to tracking")
    parser_add.add_argument("path", help="Path to file or directory")
    
    # status
    parser_status = subparsers.add_parser("status", help="Show status")
    
    # checkout
    parser_checkout = subparsers.add_parser("checkout", help="Checkout files")
    
    # remote
    parser_remote = subparsers.add_parser("remote", help="Manage remotes")
    remote_subparsers = parser_remote.add_subparsers(dest="remote_command")
    
    remote_add = remote_subparsers.add_parser("add", help="Add a remote")
    remote_add.add_argument("name", help="Remote name")
    remote_add.add_argument("type", help="Remote type (local)")
    remote_add.add_argument("path", help="Remote path")
    
    # push
    parser_push = subparsers.add_parser("push", help="Push to remote")
    
    # pull
    parser_pull = subparsers.add_parser("pull", help="Pull from remote")
    
    # repro
    parser_repro = subparsers.add_parser("repro", help="Reproduce pipeline")
    
    args = parser.parse_args()
    
    mvdvc = MvDvc()
    
    if args.command == "init":
        mvdvc.init()
    elif args.command == "add":
        mvdvc.add(args.path)
    elif args.command == "status":
        mvdvc.status()
    elif args.command == "checkout":
        mvdvc.checkout()
    elif args.command == "remote":
        if args.remote_command == "add":
            mvdvc.remote_add(args.name, args.type, args.path)
        else:
            parser_remote.print_help()
    elif args.command == "push":
        mvdvc.push()
    elif args.command == "pull":
        mvdvc.pull()
    elif args.command == "repro":
        mvdvc.repro()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
